//
// Created by hzx on 2022/9/12.
//

#include "ros/ros.h"
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <queue>
#include <mutex>
#include <opencv2/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/NavSatFix.h>
#include "pdense.h"

using namespace std;

queue<nav_msgs::Odometry> odometry_buf;
queue<sensor_msgs::ImagePtr> image_buf;
queue<sensor_msgs::NavSatFix> gps_buf;
std::mutex m_buf;

Pdense pdense("../../../src/VINS-Fusion/dense/config/dense1.yaml", "1");

void global_image_callback(const sensor_msgs::ImagePtr& global_image) {
    m_buf.lock();
    image_buf.push(global_image);
    m_buf.unlock();
}

void global_odometry_callback(const nav_msgs::Odometry& global_odomatry) {
    m_buf.lock();
    odometry_buf.push(global_odomatry);
    m_buf.unlock();
}

void global_gps_callback(const sensor_msgs::NavSatFix& global_gps) {
    m_buf.lock();
    gps_buf.push(global_gps);
    m_buf.unlock();
}

[[noreturn]] void sync_process() {
    while(true)
    {
        m_buf.lock();
        if (!odometry_buf.empty() && !image_buf.empty()) {
            double odometry_time = odometry_buf.front().header.stamp.toSec();
            double image_time = image_buf.front()->header.stamp.toSec();
            if (odometry_time < image_time - 0.003) {
                odometry_buf.pop();
                cerr << "Throw odometry." << endl;
            }
            else if (odometry_time > image_time + 0.003) {
                image_buf.pop();
                cerr << "Throw image." << endl;
            }
            else {
                Eigen::Quaterniond q;
                q.x() = odometry_buf.front().pose.pose.orientation.x;
                q.y() = odometry_buf.front().pose.pose.orientation.y;
                q.z() = odometry_buf.front().pose.pose.orientation.z;
                q.w() = odometry_buf.front().pose.pose.orientation.w;
                Eigen::Matrix3d R = q.normalized().toRotationMatrix();

                Eigen::Vector3d T;
                T.x() = odometry_buf.front().pose.pose.position.x;
                T.y() = odometry_buf.front().pose.pose.position.y;
                T.z() = odometry_buf.front().pose.pose.position.z;

                if (!pdense.get_InitGPS()) {
                    if (!gps_buf.empty() && !odometry_buf.empty()) {
                        double gps_time = gps_buf.front().header.stamp.toSec();
                        cerr << "ode: " << odometry_time << " gps: " << gps_time << endl;
                        if (odometry_time < gps_time - 0.0003) {
                            cerr << "Throw odometry(When aligning odometry and gps)." << endl;
                        }
                        else if (odometry_time > gps_time + 0.0003) {
                            gps_buf.pop();
                            cerr << "Throw gps(When aligning odometry and gps)." << endl;
                        }
                        else {
                            pdense.InitGPS(T, gps_buf.front().latitude, gps_buf.front().longitude, gps_buf.front().altitude);
                        }
                    }
                }
                cerr << "is initGPS: " << pdense.get_InitGPS() << endl;

                sensor_msgs::ImageConstPtr image_msg = image_buf.front();
                cv_bridge::CvImageConstPtr ptr;
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::TYPE_8UC3);

                cv::Mat img = ptr->image.clone();

                cout << "odometry : " << odometry_time << "  " << "image: " << image_time << endl;
//                cout << R << endl << T << endl;

                pdense.Write(R, T, img);
                odometry_buf.pop();
                image_buf.pop();
            }
        }
        m_buf.unlock();
        if (pdense.num_image_ >= pdense.max_num_) {
            pdense.pmvs2();
            break;
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "dense");
    ros::NodeHandle n("~");

    ros::Subscriber sub_global_odometry = n.subscribe("/globalEstimator/global_odometry", 100, global_odometry_callback);
    ros::Subscriber sub_global_image = n.subscribe("/vins_estimator/gps_image_0", 1000, global_image_callback);
    ros::Subscriber sub_global_gps = n.subscribe("/globalEstimator/global_gps", 100, global_gps_callback);

    std::thread sync_thread{sync_process};
    ros::spin();
    return 0;
}

