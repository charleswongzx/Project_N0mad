 
/*
 * 
 * Steering motor actuated by receiving DC voltage levels
 * Steer faster = higher voltage
 * Steer direction determined by voltage polarity
 * steerAnglePID receives cmd_steer values that represent the desired steering angle
 * steerAnglePID also receives feedback from the steering angle rotary encoder
 * steerAnglePID calculates with these and produces output based on ERROR TERM (not steering angle)
 * 
 * Brake motor also actuates by receiving DC voltage levels
 * Brake faster = higher voltage
 * The brake motor also needs a dedicated PID object
 * brakeAnglePID receives cmd_steer values that represent the desired brake pedal angle
 * brakeAnglePID also receives feedback from the brake pedal angle rotary encoder
 * brakeAnglePID calculates with these and produces output based on ERROR TERM (not brake pedal angle)
 * 
 * Since the PID library works exclusively with doubles, using doubles is necessary.
 * That said, all other variables are declared in types workable with the other nodes in the project,
 * and the declaration of doubles are kepts to a minimum. This is not a problem,
 * as doubles can be casted into floats and vice versa
 * 
 */

#include <ros.h>
#include <std_msgs/Float32.h>
#include <PID_v1.h>

ros::NodeHandle nh;

// Function Prototypes

void steerCallBack(const std_msgs::Float32 &cmdSteer);
void brakeCallBack(const std_msgs::Float32 &cmdBrake);
void throttleCallBack(const std_msgs::Float32 &cmdThrottle);

uint8_t getBrakeAngle();
uint8_t getSteerAngle();

// Pin Definitions

// TODO: DEFINE SPECIFIC PINS!!
const uint8_t steerFeedbackPin;
const uint8_t steerActuatePinForward; // Must be a PWM pin - DC voltage to steering motor will be controlled by a relay
const uint8_t steerActuatePinBackward; // Must be a PWM pin - DC voltage to steering motor will be controlled by a relay

const uint8_t brakeFeedbackPin;
const uint8_t brakeActuatePinForward; // Must be a PWM pin - DC voltage to brake motor will be controlled by a relay
const uint8_t brakeActuatePinBackward; // Must be a PWM pin - DC voltage to brake motor will be controlled by a relay

const uint8_t throttleActuatePin; // Must be a PWM pin - throttle requires 0-5V

const uint8_t brakeEncoderPinA;
const uint8_t brakeEncoderPinB;

const uint8_t steerEncoderPinA;
const uint8_t steerEncoderPinB;

uint8_t brakeEncoderAngle = 0;
uint8_t brakeEncoderAState = 0;
uint8_t brakeEncoderALastState = 0;

uint8_t steerEncoderAngle = 0;
uint8_t steerEncoderAState = 0;
uint8_t steerEncoderALastState = 0;

// PID Variable Declarations

double steerAngleDesired; // From VEHICLE_CONTROL -> cmd_steer (to be mapped into degrees)
double steerAngleFeedback; // From encoder, in degrees
double steerActuate; // To steering motor, PID'ed according to ERROR TERM

double brakeAngleDesired; // From VEHICLE_CONTROL -> cmd_brake (to be mapped into degrees)
double brakeAngleFeedback; // From encoder, in degrees
double brakeActuate; // To brake motor, PID'ed according to ERROR TERM

float throttleActuate;

// PID Parameter Declarations

// TODO: SET PID PARAMETERS!!
uint8_t steerKP; // PID parameters can be changed during run-time by SetTunings(kP, kI, kD)
uint8_t steerKI;
uint8_t steerKD;

uint8_t brakeKP;
uint8_t brakeKI;
uint8_t brakeKD;

// Input and Output Limit Declarations

// TODO: SET ALL THESE LIMITS (of PIDs and throttle value mapping)!!
float cmdSteerMin; // Check VEHICLE_CONTROL node code for these values
float cmdSteerMax; // Check VEHICLE_CONTROL node code for these values
float cmdBrakeMin; // Check VEHICLE_CONTROL node code for these values
float cmdBrakeMax; // Check VEHICLE_CONTROL node code for these values
float cmdThrottleMin; // Check VEHICLE_CONTROL node code for these values
float cmdThrottleMax; // Check VEHICLE_CONTROL node code for these values

double steerActuateMin; // Depends on relay and motor specs
double steerActuateMax; // Depends on relay and motor specs
double brakeActuateMin; // Depends on relay and motor specs
double brakeActuateMax; // Depends on relay and motor specs
float throttleActuateMin; // Depends on ESC and motor specs
float throttleActuateMax; // Depends on ESC and motor specs

PID steerAnglePID(&steerAngleFeedback, &steerActuate, &steerAngleDesired, steerKP, steerKI, steerKD, DIRECT);
PID brakeAnglePID(&brakeAngleFeedback, &brakeActuate, &brakeAngleDesired, brakeKP, brakeKI, brakeKD, DIRECT);

void steerCallBack(const std_msgs::Float32 &cmdSteer) {
  
  steerAngleDesired = cmdSteer.data;
  if (steerActuate >= 0) {
    analogWrite(steerActuatePinForward, steerActuate); // Send signal through relays governing current flow for forward motor rotation (H-bridge)
    analogWrite(steerActuatePinBackward, 0); // Close relays governing current flow for backward motor rotation (H-bridge)
  } else {
    analogWrite(steerActuatePinBackward, -steerActuate); // Send signal through relays governing current flow for backward motor rotation (H-bridge)
    analogWrite(steerActuatePinForward, 0); // Close relays governing current flow for forward motor rotation (H-bridge)
  
  }
}

void brakeCallBack(const std_msgs::Float32 &cmdBrake) {
  
  brakeAngleDesired = cmdBrake.data;
 
  if (brakeActuate >= 0) {
    analogWrite(brakeActuatePinForward, &brakeActuate); // Send signal through relays governing current flow for forward motor rotation (H-bridge)
    analogWrite(brakeActuatePinBackward, 0); // Close relays governing current flow for backward motor rotation (H-bridge)
  } else {
    analogWrite(brakeActuatePinBackward, -brakeActuate); // Send signal through relays governing current flow for backward motor rotation (H-bridge)
    analogWrite(brakeActuatePinForward, 0); // Close relays governing current flow for forward motor rotation (H-bridge)
  }
  
}

void throttleCallBack(const std_msgs::Float32 &cmdThrottle) {
  
  throttleActuate = map(cmdThrottle.data, cmdThrottleMin, cmdThrottleMax, throttleActuateMin, throttleActuateMax);
  analogWrite(throttleActuatePin, throttleActuate);
  
}

ros::Subscriber<std_msgs::Float32> steerSubscriber("cmd_steer", &steerCallBack);
ros::Subscriber<std_msgs::Float32> brakeSubscriber("cmd_brake", &brakeCallBack);
ros::Subscriber<std_msgs::Float32> throttleSubscriber("cmd_throttle", &throttleCallBack);

void setup() {
  
  pinMode(steerFeedbackPin, INPUT);
  pinMode(steerActuatePinForward, OUTPUT);
  pinMode(steerActuatePinBackward, OUTPUT);
  
  pinMode(brakeFeedbackPin, INPUT);
  pinMode(brakeActuatePinForward, OUTPUT);
  pinMode(brakeActuatePinBackward, OUTPUT);
  
  pinMode(throttleActuatePin, OUTPUT);
  
  pinMode(brakeEncoderPinA, INPUT);
  pinMode(brakeEncoderPinB, INPUT);
  
  pinMode(steerEncoderPinA, INPUT);
  pinMode(steerEncoderPinB, INPUT);

  steerAnglePID.SetMode(AUTOMATIC); // Starts PID
  brakeAnglePID.SetMode(AUTOMATIC); // Starts PID
  steerAnglePID.SetOutputLimits(steerActuateMin, steerActuateMax);
  brakeAnglePID.SetOutputLimits(brakeActuateMin, brakeActuateMax);

  brakeEncoderALastState = digitalRead(brakeEncoderPinA);
  steerEncoderALastState = digitalRead(steerEncoderPinA);

}

void loop() {
  
  steerAngleFeedback = getSteerAngle(); // Every clock, read encoder
  brakeAngleFeedback = getBrakeAngle(); // Every clock, read encoder

  steerAnglePID.Compute(); // Every clock, run PID algorithm
  brakeAnglePID.Compute(); // Every clock, run PID algorithm
  
  nh.spinOnce(); // Every clock, check for messages and trigger callbacks
  delay(1); // Optimal delay between clocks
  
}

uint8_t getBrakeAngle() { // Read brake motor rotary encoder
  
  brakeEncoderAState = digitalRead(brakeEncoderPinA);
  
  if (digitalRead(brakeEncoderPinB) != brakeEncoderAState) {
    brakeEncoderAngle ++;
  } else {
    brakeEncoderAngle --;
  }

  brakeEncoderALastState = brakeEncoderAState;
  
}

uint8_t getSteerAngle() { // Read steering motor rotary encoder
  
  steerEncoderAState = digitalRead(steerEncoderPinA);
  
  if (digitalRead(steerEncoderPinB) != steerEncoderAState) {
    steerEncoderAngle ++;
  } else {
    steerEncoderAngle --;
  }

  steerEncoderALastState = steerEncoderAState;
  
}

