/*
 * playback_verify.ino
 *
 * Level C of the Python-to-AVR equivalence check. Receives 10-bit ADC codes
 * over serial, runs the deployed DSP and classifier, and reports staged
 * diagnostics so each pipeline stage can be compared against the float32
 * reference in embedded_reference.py.
 *
 * This is deliberately NOT a real-time test. The host paces the transfer with
 * a chunked handshake, so serial throughput never limits or perturbs the
 * arithmetic. Timing is measured separately by deployability_profile.ino.
 *
 * Filter: transposed direct form II, matching scipy.signal.sosfilt and the
 * float32 reference. a0 is assumed to be 1 and is not read.
 *
 * State is fully reset on every BEGIN, which is why the host streams one
 * trial per file rather than a concatenation.
 *
 * PROTOCOL
 *   host -> avr   "BEGIN <n_samples> <mode>\n"   mode 0 = SUMMARY, 1 = FILTER
 *   avr  -> host  REC_READY
 *   host -> avr   raw binary, n_samples * 2 int16 little endian, ch0 ch1
 *                 delivered in chunks of CHUNK_SAMPLES
 *   avr  -> host  REC_FILT after each chunk   (mode 1 only)
 *                 REC_WIN  as each window completes
 *                 REC_ACK  after each chunk
 *   avr  -> host  REC_DONE
 *
 * Every avr -> host message is a binary record:
 *   0xA5 0x5A | type u8 | length u16 LE | payload
 */

#include "playback_params.h"

#define MODE_SUMMARY 0
#define MODE_FILTER  1

#define REC_READY 0x01
#define REC_ACK   0x02
#define REC_FILT  0x03
#define REC_WIN   0x04
#define REC_DONE  0x05
#define REC_INFO  0x06
#define REC_ERR   0x07

#define CHUNK_SAMPLES 8           /* 8 samples x 2 ch x 2 B = 32 B, well under
                                     the 64 B hardware serial receive buffer.
                                     rx[] is sized from this, so the host must
                                     send exactly this many samples per chunk.
                                     run_playback.py enforces the match. */

/* ------------------------------------------------------------ state */

struct SubBlock {
  float   sum_sq;
  float   sum_abs;
  float   wl;
  float   first;
  float   last;
  uint8_t zc;                     /* max 99 crossings in SUB_N samples */
};

static SubBlock blocks[N_CH][2];  /* two in rotation per channel */
static uint8_t  cur_blk   = 0;    /* index currently accumulating */
static bool     have_prev = false;
static uint16_t blk_count = 0;    /* samples pushed into the current block */

static float    z[N_SECTIONS][N_CH][2];   /* TDF-II filter state */

static uint32_t sample_idx = 0;   /* index into the trial, pre-transient */
static uint32_t n_expected = 0;
static uint8_t  mode       = MODE_SUMMARY;

static int16_t  rx[CHUNK_SAMPLES * N_CH];
static float    filt_out[CHUNK_SAMPLES * N_CH];

/* ------------------------------------------------------------ records */

static void send_record(uint8_t type, const void *payload, uint16_t len) {
  Serial.write(0xA5);
  Serial.write(0x5A);
  Serial.write(type);
  Serial.write((uint8_t)(len & 0xFF));
  Serial.write((uint8_t)(len >> 8));
  if (len) Serial.write((const uint8_t *)payload, len);
  Serial.flush();
}

static void send_info(const char *s) {
  send_record(REC_INFO, s, strlen(s));
}

/* ------------------------------------------------------------ crc */

/* CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor.
   Computed over the raw received bytes so the host can confirm the ADC codes
   arrived intact. Without this, a dropped serial byte is indistinguishable
   from an implementation error. */
static uint16_t rx_crc;

static uint16_t crc16_update(uint16_t crc, uint8_t b) {
  crc ^= (uint16_t)b << 8;
  for (uint8_t i = 0; i < 8; i++)
    crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  return crc;
}

/* ------------------------------------------------------------ helpers */

static inline float sgn(float x) { return (x >= 0.0f) ? 1.0f : -1.0f; }

static void reset_block(SubBlock *b) {
  b->sum_sq  = 0.0f;
  b->sum_abs = 0.0f;
  b->wl      = 0.0f;
  b->first   = 0.0f;
  b->last    = 0.0f;
  b->zc      = 0;
}

static void reset_all(void) {
  for (uint8_t s = 0; s < N_SECTIONS; s++)
    for (uint8_t c = 0; c < N_CH; c++) {
      z[s][c][0] = 0.0f;
      z[s][c][1] = 0.0f;
    }
  for (uint8_t c = 0; c < N_CH; c++) {
    reset_block(&blocks[c][0]);
    reset_block(&blocks[c][1]);
  }
  cur_blk    = 0;
  have_prev  = false;
  blk_count  = 0;
  sample_idx = 0;
  rx_crc     = 0xFFFF;
}

/* Transposed direct form II, one sample through all sections.
   a0 is assumed 1 and is not read. */
static float filter_sample(uint8_t ch, float v) {
  for (uint8_t s = 0; s < N_SECTIONS; s++) {
    const float b0 = pgm_read_float(&SOS[s * 6 + 0]);
    const float b1 = pgm_read_float(&SOS[s * 6 + 1]);
    const float b2 = pgm_read_float(&SOS[s * 6 + 2]);
    const float a1 = pgm_read_float(&SOS[s * 6 + 4]);
    const float a2 = pgm_read_float(&SOS[s * 6 + 5]);

    const float y = b0 * v + z[s][ch][0];
    z[s][ch][0] = b1 * v - a1 * y + z[s][ch][1];
    z[s][ch][1] = b2 * v - a2 * y;
    v = y;
  }
  return v;
}

/* Accumulate one filtered sample into the current sub-block. Self-contained:
   no term involving a neighbouring block is added here. */
static void block_push(uint8_t ch, float x) {
  SubBlock *b = &blocks[ch][cur_blk];
  b->sum_sq  += x * x;
  b->sum_abs += fabsf(x);
  if (blk_count == 0) {
    b->first = x;
  } else {
    const float d = x - b->last;
    b->wl += fabsf(d);
    if (sgn(b->last) != sgn(x) && fabsf(d) >= ZC_EPS) b->zc++;
  }
  b->last = x;
}

/* Join two adjacent sub-blocks. Adds the single a.last -> b.first term that
   neither block owns. */
static void combine(const SubBlock *a, const SubBlock *b, float *out4) {
  const float n = (float)WIN_N;
  const float d = b->first - a->last;

  out4[0] = sqrtf((a->sum_sq + b->sum_sq) / n);          /* RMS */
  out4[1] = (a->sum_abs + b->sum_abs) / n;               /* MAV */
  out4[2] = (a->wl + b->wl + fabsf(d)) / n;              /* WL  */

  uint16_t zc = (uint16_t)a->zc + (uint16_t)b->zc;
  if (sgn(a->last) != sgn(b->first) && fabsf(d) >= ZC_EPS) zc++;
  out4[3] = (float)zc / (n / FS);                        /* ZC, per second */
}

/* Emit one window: 8 raw features, 8 scaled, 3 scores, 1 prediction. */
static void emit_window(void) {
  float raw[N_FEAT];
  for (uint8_t c = 0; c < N_CH; c++)
    combine(&blocks[c][1 - cur_blk], &blocks[c][cur_blk], &raw[c * 4]);

  float scaled[N_FEAT];
  for (uint8_t f = 0; f < N_FEAT; f++) {
    const float m  = pgm_read_float(&SCALER_MEAN[f]);
    const float iv = pgm_read_float(&SCALER_INV_SCALE[f]);
    scaled[f] = (raw[f] - m) * iv;
  }

  float score[N_CLASS];
  for (uint8_t c = 0; c < N_CLASS; c++) {
    float acc = pgm_read_float(&LDA_INTERCEPT[c]);
    for (uint8_t f = 0; f < N_FEAT; f++)
      acc += pgm_read_float(&LDA_COEF[c * N_FEAT + f]) * scaled[f];
    score[c] = acc;
  }

  uint8_t pred = 0;
  for (uint8_t c = 1; c < N_CLASS; c++)
    if (score[c] > score[pred]) pred = c;

  uint8_t buf[(N_FEAT + N_FEAT + N_CLASS) * 4 + 1];
  uint16_t o = 0;
  memcpy(buf + o, raw,    N_FEAT  * 4); o += N_FEAT  * 4;
  memcpy(buf + o, scaled, N_FEAT  * 4); o += N_FEAT  * 4;
  memcpy(buf + o, score,  N_CLASS * 4); o += N_CLASS * 4;
  buf[o++] = pred;
  send_record(REC_WIN, buf, o);
}

/* ------------------------------------------------------------ chunk */

static void process_chunk(uint16_t n_samp) {
  for (uint16_t i = 0; i < n_samp; i++) {
    for (uint8_t c = 0; c < N_CH; c++) {
      const float v = (float)rx[i * N_CH + c] * ADC_LSB - V_MAX;
      const float y = filter_sample(c, v);
      if (mode == MODE_FILTER) filt_out[i * N_CH + c] = y;

      if (sample_idx >= TRANSIENT_N) block_push(c, y);
    }

    if (sample_idx >= TRANSIENT_N) {
      blk_count++;
      if (blk_count == SUB_N) {
        if (have_prev) emit_window();
        have_prev = true;
        cur_blk   = 1 - cur_blk;
        for (uint8_t c = 0; c < N_CH; c++) reset_block(&blocks[c][cur_blk]);
        blk_count = 0;
      }
    }
    sample_idx++;
  }

  if (mode == MODE_FILTER)
    send_record(REC_FILT, filt_out, n_samp * N_CH * 4);
}

/* ------------------------------------------------------------ arduino */

/* Reported on every BEGIN rather than once at boot: the host resets its input
   buffer before each command, so a boot-time message would be discarded. */
static void send_sizes(void) {
  char msg[80];
  snprintf(msg, sizeof(msg),
           "sizeof(SubBlock)=%u accum=%u filter=%u",
           (unsigned)sizeof(SubBlock),
           (unsigned)sizeof(blocks),
           (unsigned)sizeof(z));
  send_info(msg);
}

void setup(void) {
  Serial.begin(115200);
  while (!Serial) { }
  Serial.setTimeout(5000);
}

void loop(void) {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (!line.startsWith("BEGIN")) return;

  int sp1 = line.indexOf(' ');
  int sp2 = line.indexOf(' ', sp1 + 1);
  if (sp1 < 0 || sp2 < 0) { send_record(REC_ERR, "bad BEGIN", 9); return; }

  n_expected = (uint32_t)line.substring(sp1 + 1, sp2).toInt();
  mode       = (uint8_t)line.substring(sp2 + 1).toInt();

  reset_all();
  send_sizes();
  send_record(REC_READY, NULL, 0);

  uint32_t done = 0;
  while (done < n_expected) {
    uint16_t want = CHUNK_SAMPLES;
    if (n_expected - done < want) want = (uint16_t)(n_expected - done);

    const size_t need = (size_t)want * N_CH * 2;
    const size_t got  = Serial.readBytes((char *)rx, need);
    if (got != need) { send_record(REC_ERR, "timeout", 7); return; }

    const uint8_t *raw = (const uint8_t *)rx;
    for (size_t k = 0; k < need; k++) rx_crc = crc16_update(rx_crc, raw[k]);

    process_chunk(want);
    done += want;
    send_record(REC_ACK, NULL, 0);
  }

  /* DONE carries the CRC of every ADC byte received, little endian. */
  uint8_t crcbuf[2] = { (uint8_t)(rx_crc & 0xFF), (uint8_t)(rx_crc >> 8) };
  send_record(REC_DONE, crcbuf, 2);
}
