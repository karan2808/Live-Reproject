#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <ctime>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <fstream>
#include <iterator>
// Include VelodyneCapture Header
#include "VelodyneCapture.h"

using namespace std;
using namespace cv;

string current_dt;

string Get_DateTime(){
    time_t rawtime;
    timeval currentTime;
    struct tm *timeinfo;

    char buffer[80];
    gettimeofday(&currentTime, NULL);
    int milli = currentTime.tv_usec / 1000;

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%d_%m_%Y_%H_%M_%S", timeinfo);

    char Current_Time[84] = "";
    sprintf(Current_Time, "%s_%03d", buffer, milli);
    return Current_Time;
}

int main( int argc, char* argv[] )
{
    // Open VelodyneCapture that retrieve from Sensor
    const boost::asio::ip::address address = boost::asio::ip::address::from_string( "192.168.1.21" );
    const unsigned short port = 2368;
    velodyne::VLP16Capture capture( address, port );
    //velodyne::HDL32ECapture capture( address, port );

    /*
    // Open VelodyneCapture that retrieve from PCAP
    const std::string filename = "../file.pcap";
    velodyne::VLP16Capture capture( filename );
    //velodyne::HDL32ECapture capture( filename );
    */

    // make a camera capture object
    Mat frame;
    VideoCapture cap;
    int deviceId = 0; // open the default camera
    int apiId = cv::CAP_ANY;
    cap.open(deviceId, apiId);

    if( !capture.isOpen() ){
        std::cerr << "Can't open VelodyneCapture." << std::endl;
        return -1;
    }

    // Create Viewer
    cv::viz::Viz3d viewer( "Velodyne" );

    // Register Keyboard Callback
    viewer.registerKeyboardCallback(
        []( const cv::viz::KeyboardEvent& event, void* cookie ){
            // Close Viewer
            if( event.code == 'q' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN ){
                static_cast<cv::viz::Viz3d*>( cookie )->close();
            }
        }
        , &viewer
    );

    while( capture.isRun() && !viewer.wasStopped() ){
        // Capture One Rotation Data
        std::vector<velodyne::Laser> lasers;
        capture >> lasers;
        if( lasers.empty() ){
            continue;
        }
        
        // make a vector of strings to append to the file
        std::vector<std::string> vecOfStr;

        // Convert to 3-dimention Coordinates
        std::vector<cv::Vec3f> buffer( lasers.size() );
        for( const velodyne::Laser& laser : lasers ){
            const double distance = static_cast<double>( laser.distance );
            const double azimuth  = laser.azimuth  * CV_PI / 180.0;
            const double vertical = laser.vertical * CV_PI / 180.0;
            float x = static_cast<float>( ( distance * std::cos( vertical ) ) * std::sin( azimuth ) );
            float y = static_cast<float>( ( distance * std::cos( vertical ) ) * std::cos( azimuth ) );
            float z = static_cast<float>( ( distance * std::sin( vertical ) ) );

            if( x == 0.0f && y == 0.0f && z == 0.0f ){
                x = std::numeric_limits<float>::quiet_NaN();
                y = std::numeric_limits<float>::quiet_NaN();
                z = std::numeric_limits<float>::quiet_NaN();
            }
            string newline;
            if (!isnan(x)){
                // file << x << " " << y << " " << z << " " << std::endl;
                newline = to_string(x) + " " + to_string(y) + " " + to_string(z);
                //cout << newline << endl;
                vecOfStr.push_back(newline);
            }
            // buffer.push_back( cv::Vec3f( x, y, z ) );
        }

        // get the current time stamp
        string current_time = Get_DateTime();

        // open a file with current time
        ofstream file;
        file.open(("data/" + current_time.substr(0, 21) +".txt").c_str(), std::ofstream::out);
        std::ostream_iterator<std::string> output_iterator(file, "\n");
        std::copy(vecOfStr.begin(), vecOfStr.end(), output_iterator);
        // close the file
        file.close();

        // capture the image frame and save it
        cap.read(frame);
        imwrite(("data/" + current_time.substr(0, 21) +".tiff").c_str(), frame);
        // string latest_time = Get_DateTime();
        // rename((("data/" + current_time +".txt").c_str()), (("data/" + latest_time +".txt").c_str()));
        // Create Widget
        // cv::Mat cloudMat = cv::Mat( static_cast<int>( buffer.size() ), 1, CV_32FC3, &buffer[0] );
        // cv::viz::WCloud cloud( cloudMat );

        // // Show Point Cloud
        // viewer.showWidget( "Cloud", cloud );
        // viewer.spinOnce();
    }

    // Close All Viewers
    cv::viz::unregisterAllWindows();

    return 0;
}
