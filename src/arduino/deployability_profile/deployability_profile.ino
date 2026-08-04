#include <avr/pgmspace.h>
#include <math.h>

const int NUM_FEATURES = 8;
const int NUM_CLASSES = 3;
const int SAMPLES_PER_BLOCK = 100;

const float ZC_DEADZONE_COUNTS = 30.7; 
const float INV_SAMPLES_WINDOW = 1.0 / (SAMPLES_PER_BLOCK * 2.0);
const float INV_ZC_TIME = 1.0 / 0.2; 

volatile float volatile_sink = 0;

const float lda_weights[NUM_CLASSES][NUM_FEATURES] PROGMEM = {
  {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08},
  {0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18},
  {0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28}
};
const float lda_intercepts[NUM_CLASSES] PROGMEM = {0.1, 0.2, 0.3};
const float scaler_mean[NUM_FEATURES] PROGMEM = {0,0,0,0,0,0,0,0};
const float scaler_inv_scale[NUM_FEATURES] PROGMEM = {1,1,1,1,1,1,1,1}; 

const float b0_1 = 0.8, b1_1 = -1.6, b2_1 = 0.8, a1_1 = -1.5, a2_1 = 0.6;
const float b0_2 = 1.0, b1_2 = -2.0, b2_2 = 1.0, a1_2 = -1.4, a2_2 = 0.5;

float x1_state[2][2]={0}, x2_state[2][2]={0}, y1_state[2][2]={0}, y2_state[2][2]={0};

float rms_accum[2]={0}, mav_accum[2]={0}, wl_accum[2]={0};
int zc_accum[2]={0};

float rms_prev[2]={0}, mav_prev[2]={0}, wl_prev[2]={0};
int zc_prev[2]={0};

float prev_val[2]={0};
float finalized_features[8] = {0};
float scaled_features[8] = {0}; 

void set_adc_prescaler(uint8_t mode) {
  ADCSRA = (ADCSRA & 0xf8) | mode; 
}

void setup() {
  Serial.begin(115200);
  while(!Serial);
  delay(100);

  Serial.println("\n--- Deployability Profiling ---");
  unsigned long start, end;
  float dt;

  // 0. micros() call overhead
  start = micros();
  end = micros();
  Serial.print("micros() call overhead: "); Serial.print(end - start); Serial.println(" us");

  // 1. Empty Loop Overhead
  start = micros();
  for(int i=0; i<1000; i++) {
    __asm__ __volatile__ ("nop\n\t"); 
  }
  end = micros();
  float loop_overhead = (end - start) / 1000.0;
  Serial.print("Empty loop overhead: "); Serial.print(loop_overhead); Serial.println(" us");

  // 2a. ADC Latency (Default Prescaler 128)
  set_adc_prescaler(7);
  start = micros();
  for(int i=0; i<1000; i++) {
    volatile int a = analogRead(A0);
    volatile int b = analogRead(A1);
  }
  end = micros();
  dt = ((end - start) / 1000.0) - loop_overhead;
  Serial.print("ADC (2 ch, PS=128): "); Serial.print(dt); Serial.println(" us");

  // 2b. ADC Latency (Fast Prescaler 16)
  set_adc_prescaler(4);
  start = micros();
  for(int i=0; i<1000; i++) {
    volatile int a = analogRead(A0);
    volatile int b = analogRead(A1);
  }
  end = micros();
  dt = ((end - start) / 1000.0) - loop_overhead;
  Serial.print("ADC (2 ch, PS=16): "); Serial.print(dt); Serial.println(" us");

  // Reset ADC prescaler back to 128 for remaining tests
  set_adc_prescaler(7);

  // 3. Filter & Accumulation
  start = micros();
  for(int i=0; i<1000; i++) {
    for(int ch=0; ch<2; ch++) {
      float input = (float)(i % 512) - 256.0; 
      
      float out1 = b0_1*input + b1_1*x1_state[ch][0] + b2_1*x2_state[ch][0] - a1_1*y1_state[ch][0] - a2_1*y2_state[ch][0];
      x2_state[ch][0] = x1_state[ch][0]; x1_state[ch][0] = input;
      y2_state[ch][0] = y1_state[ch][0]; y1_state[ch][0] = out1;
      
      float final_out = b0_2*out1 + b1_2*x1_state[ch][1] + b2_2*x2_state[ch][1] - a1_2*y1_state[ch][1] - a2_2*y2_state[ch][1];
      x2_state[ch][1] = x1_state[ch][1]; x1_state[ch][1] = out1;
      y2_state[ch][1] = y1_state[ch][1]; y1_state[ch][1] = final_out;

      rms_accum[ch] += final_out * final_out;
      mav_accum[ch] += fabs(final_out);
      wl_accum[ch] += fabs(final_out - prev_val[ch]);
      
      if ((final_out * prev_val[ch] < 0) && (fabs(final_out - prev_val[ch]) >= ZC_DEADZONE_COUNTS)) {
        zc_accum[ch]++;
      }
      prev_val[ch] = final_out;
    }
  }
  end = micros();
  dt = ((end - start) / 1000.0) - loop_overhead;
  Serial.print("Filter + Accumulation (2 ch): "); Serial.print(dt); Serial.println(" us");

  // 4. Feature Finalization
  start = micros();
  for(int i=0; i<1000; i++) {
    for(int ch=0; ch<2; ch++) {
      float rms_tot = rms_accum[ch] + rms_prev[ch];
      float mav_tot = mav_accum[ch] + mav_prev[ch];
      float wl_tot  = wl_accum[ch] + wl_prev[ch];
      int zc_tot    = zc_accum[ch] + zc_prev[ch];

      finalized_features[ch*4 + 0] = sqrt(rms_tot * INV_SAMPLES_WINDOW);
      finalized_features[ch*4 + 1] = mav_tot * INV_SAMPLES_WINDOW;
      finalized_features[ch*4 + 2] = wl_tot * INV_SAMPLES_WINDOW;
      finalized_features[ch*4 + 3] = (float)zc_tot * INV_ZC_TIME;

      rms_prev[ch] = rms_accum[ch]; mav_prev[ch] = mav_accum[ch];
      wl_prev[ch] = wl_accum[ch]; zc_prev[ch] = zc_accum[ch];
      
      volatile_sink += finalized_features[ch*4]; 
    }
  }
  end = micros();
  dt = ((end - start) / 1000.0) - loop_overhead;
  Serial.print("Sub-block Combine + Finalization: "); Serial.print(dt); Serial.println(" us");

  // 5. Inference
  start = micros();
  for(int i=0; i<1000; i++) { 
    float scores[3] = {0, 0, 0};
    
    for(int f=0; f<NUM_FEATURES; f++) {
      float mean = pgm_read_float(&scaler_mean[f]);
      float scale_inv = pgm_read_float(&scaler_inv_scale[f]);
      scaled_features[f] = (finalized_features[f] - mean) * scale_inv;
    }

    for(int c=0; c<NUM_CLASSES; c++) {
      scores[c] = pgm_read_float(&lda_intercepts[c]);
      for(int f=0; f<NUM_FEATURES; f++) {
        scores[c] += scaled_features[f] * pgm_read_float(&lda_weights[c][f]);
      }
    }
    volatile_sink += scores[0] + scores[1] + scores[2];
  }
  end = micros();
  dt = ((end - start) / 1000.0) - loop_overhead;
  Serial.print("LDA Inference (Scale + MM): "); Serial.print(dt); Serial.println(" us");

  // 6. WCET Pipeline Tick (At prescaler 128)
  unsigned long max_tick = 0;
  for(int i=0; i<1000; i++) {
    unsigned long tick_start = micros();
    
    volatile int a = analogRead(A0);
    volatile int b = analogRead(A1);

    for(int ch=0; ch<2; ch++) {
      float input = (float)(a); 
      
      float out1 = b0_1*input + b1_1*x1_state[ch][0] + b2_1*x2_state[ch][0] - a1_1*y1_state[ch][0] - a2_1*y2_state[ch][0];
      x2_state[ch][0] = x1_state[ch][0]; x1_state[ch][0] = input;
      y2_state[ch][0] = y1_state[ch][0]; y1_state[ch][0] = out1;
      
      float final_out = b0_2*out1 + b1_2*x1_state[ch][1] + b2_2*x2_state[ch][1] - a1_2*y1_state[ch][1] - a2_2*y2_state[ch][1];
      x2_state[ch][1] = x1_state[ch][1]; x1_state[ch][1] = out1;
      y2_state[ch][1] = y1_state[ch][1]; y1_state[ch][1] = final_out;
      
      rms_accum[ch] += final_out * final_out;
      mav_accum[ch] += fabs(final_out);
      wl_accum[ch] += fabs(final_out - prev_val[ch]);
      
      if ((final_out * prev_val[ch] < 0) && (fabs(final_out - prev_val[ch]) >= ZC_DEADZONE_COUNTS)) {
        zc_accum[ch]++;
      }
      prev_val[ch] = final_out;

      float rms_tot = rms_accum[ch] + rms_prev[ch];
      float mav_tot = mav_accum[ch] + mav_prev[ch];
      float wl_tot  = wl_accum[ch] + wl_prev[ch];
      int zc_tot    = zc_accum[ch] + zc_prev[ch];

      finalized_features[ch*4 + 0] = sqrt(rms_tot * INV_SAMPLES_WINDOW);
      finalized_features[ch*4 + 1] = mav_tot * INV_SAMPLES_WINDOW;
      finalized_features[ch*4 + 2] = wl_tot * INV_SAMPLES_WINDOW;
      finalized_features[ch*4 + 3] = (float)zc_tot * INV_ZC_TIME;

      rms_prev[ch] = rms_accum[ch]; mav_prev[ch] = mav_accum[ch];
      wl_prev[ch] = wl_accum[ch]; zc_prev[ch] = zc_accum[ch];
    }

    for(int f=0; f<NUM_FEATURES; f++) {
      scaled_features[f] = (finalized_features[f] - pgm_read_float(&scaler_mean[f])) * pgm_read_float(&scaler_inv_scale[f]);
    }
    float scores[3] = {0,0,0};
    for(int c=0; c<NUM_CLASSES; c++) {
      scores[c] = pgm_read_float(&lda_intercepts[c]);
      for(int f=0; f<NUM_FEATURES; f++) {
        scores[c] += scaled_features[f] * pgm_read_float(&lda_weights[c][f]);
      }
    }
    volatile_sink += scores[0];

    unsigned long tick_duration = micros() - tick_start;
    if(tick_duration > max_tick) {
      max_tick = tick_duration;
    }
  }
  Serial.print("WCET (Max Full Pipeline Tick): "); Serial.print(max_tick); Serial.println(" us");
}

void loop() {}