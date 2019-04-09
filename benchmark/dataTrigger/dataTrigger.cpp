#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <iostream>

int main(void) {
    cluon::OD4Session od4{253};
    const int ellipseID = 112;

    // Function to call on newly arriving Envelopes which contain accelerometer data.
    long Ts = cluon::time::toMicroseconds(cluon::time::now());
    auto accRecived{
        [&ellipseID] (cluon::data::Envelope &&envelope) {
            if (envelope.senderStamp() == ellipseID) {
                std::cout << "acc" << std::endl;
                opendlv::proxy::AccelerationReading accData = cluon::extractMessage<opendlv::proxy::AccelerationReading>(std::move(envelope));
                cluon::data::TimeStamp ts = envelope.sampleTimeStamp();
                long timestamp = cluon::time::toMicroseconds(ts);
                float accX = accData.accelerationX();
                float accY = accData.accelerationY();
                float accZ = accData.accelerationZ();
                // pVISLAM->collectImuData(cfsd::SensorType::ACCELEROMETER, timestamp, accX, accY, accZ);
            }
        }
    };

    // Function to call on newly arriving Envelopes which contain gyroscope data.
    auto gyrRecived{
        [&ellipseID, &Ts] (cluon::data::Envelope &&envelope) {
            long lastTs = Ts;
            Ts = cluon::time::toMicroseconds(cluon::time::now());
            if (envelope.senderStamp() == ellipseID) {
                std::cout << "gyr" 
                          << "(elapsed: " << Ts - lastTs << "us)" << std::endl;
                opendlv::proxy::AngularVelocityReading gyrData = cluon::extractMessage<opendlv::proxy::AngularVelocityReading>(std::move(envelope));
                cluon::data::TimeStamp ts = envelope.sampleTimeStamp();
                long timestamp = cluon::time::toMicroseconds(ts);
                float gyrX = gyrData.angularVelocityX();
                float gyrY = gyrData.angularVelocityY();
                float gyrZ = gyrData.angularVelocityZ();
                // pVISLAM->collectImuData(cfsd::SensorType::GYROSCOPE, timestamp, gyrX, gyrY, gyrZ);
            }
        }
    };

    // Set a delegate to be called data-triggered on arrival of a new Envelope for a given message identifier.
    od4.dataTrigger(opendlv::proxy::AccelerationReading::ID(), accRecived);
    od4.dataTrigger(opendlv::proxy::AngularVelocityReading::ID(), gyrRecived);

    while(od4.isRunning()) {
        
    }

    return 0;
}