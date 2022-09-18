//
// Created by hzx on 2022/7/11.
//

#include "pdense.h"
#include <sys/stat.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/parse.h>
#include <pcl/io/vtk_io.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/conversions.h>
#include <iostream>

#include "../base/pmvs/findMatch.h"

using namespace std;

Pdense::Pdense(const string &config_path) {
    cv::FileStorage fsSettings(config_path, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
        cerr << "ERROR: Wrong path to settings" << endl;

    config_path_ = config_path;

    fsSettings["option_name"] >> option_name_;
    fsSettings["max_num"] >> max_num_;
    fsSettings["is_PCD"] >> isPCD_;
    fsSettings["is_global"] >> isGlobal_;

    string path, root_name;
    fsSettings["path"] >> path;
    fsSettings["root_name"] >> root_name;
    root_path_ = path + root_name + "/";

    string folder_path = path + "pdense";
    if(access(root_path_.c_str(), 0)){
        cerr << "Folder(" + root_name + ") does not exist! Will create a new one!" << endl;
        if(mkdir(root_path_.c_str(), 0771) == 0){ // 成功返回0，不成功返回-1
            cout << "Folder(" + root_name + ") created successfully" << endl;
            mkdir(string(root_path_ + "models").c_str(), 0771);
            mkdir(string(root_path_ + "txt").c_str(), 0771);
            mkdir(string(root_path_ + "visualize").c_str(), 0771);
        }
    }
    else {
        cerr << "Folder(" + root_name + ") exist!" << endl;
    }
}

Pdense::Pdense(const string &config_path, const string& option_name) : Pdense(config_path) {
    option_name_ = option_name;
}


void Pdense::Write(const Eigen::Matrix3d& R, const Eigen::Vector3d& T, const cv::Mat& img) {
    if (!get_InitGPS())
        return;
    stringstream name;
    name << setfill('0') << setw(8) << num_image_;
    ofstream foutC;
    foutC.open(root_path_ + "txt/" + name.str() + ".txt", std::ios::out);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
    if (!foutC.is_open()) {
        cerr << "Projection storage failed for picture number " << to_string(num_image_) << "." << endl;
        return;
    }

//    Eigen::Matrix3d tmp_R = estimator.Rs[WINDOW_SIZE] * estimator.ric[0];
//    Eigen::Vector3d tmp_T = estimator.Ps[WINDOW_SIZE] + estimator.Rs[WINDOW_SIZE] * estimator.tic[0];
    Eigen::Matrix3d tmp_R = R;
    Eigen::Vector3d tmp_T = T;
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;
    ric << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    tic << 0, 0 ,0;
    Eigen::Matrix3d K;
//    Eigen::Matrix3d tmp_R = R * ric;
//    Eigen::Vector3d tmp_T = T + R * tic;

    /* kitti_10_03_config.yaml */
    K << 718.856 , 0 , 607.1928,
            0 , 718.856 , 185.2157,
            0, 0, 1;
//    /* cam0_pinhole.yaml */
//    K << 461.15862106007575 , 0 , 362.65929181685937,
//            0 , 459.75286598073296 , 248.52105668448124,
//            0, 0, 1;
    Eigen::Matrix<double, 3, 4> projection;
    tmp_R = tmp_R.inverse().eval();
    tmp_T = tmp_R * tmp_T;

    projection << tmp_R(0,0) , tmp_R(0,1) , tmp_R(0,2) , -tmp_T(0),
            tmp_R(1,0) , tmp_R(1,1) , tmp_R(1,2) , -tmp_T(1),
            tmp_R(2,0) , tmp_R(2,1) , tmp_R(2,2) , -tmp_T(2);
    projection = K * projection;

    foutC << "CONTOUR" << endl;
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(16);
    foutC << projection(0,0) << " " << projection(0,1) << " " << projection(0,2) << " " << projection(0,3) << endl
          << projection(1,0) << " " << projection(1,1) << " " << projection(1,2) << " " << projection(1,3) << endl
          << projection(2,0) << " " << projection(2,1) << " " << projection(2,2) << " " << projection(2,3) << endl;

    foutC.close();

    if (img.channels() <= 1)
        cv::imwrite(root_path_+ "visualize/" + name.str() + ".jpg", Gray2Color(img));
    else
        cv::imwrite(root_path_+ "visualize/" + name.str() + ".jpg", img);

    std::ofstream foutC2("/home/hzx/catkin_ws/vio_estimator_" + option_name_ + ".csv", ios::app);
    foutC2.setf(ios::fixed, ios::floatfield);
    foutC2.precision(16);
    foutC2 << tmp_T.x() << ","
           << tmp_T.y() << ","
           << tmp_T.z() << endl;

    foutC2.close();

    ++num_image_;
}

cv::Mat Pdense::Gray2Color(const cv::Mat& phase)
{
    CV_Assert(phase.channels() == 1);

    cv::Mat temp, result, mask;
    // 将灰度图重新归一化至0-255
    cv::normalize(phase, temp, 255, 0, cv::NORM_MINMAX);
    temp.convertTo(temp, CV_8UC1);
    // 创建掩膜，目的是为了隔离nan值的干扰
    mask = cv::Mat::zeros(phase.size(), CV_8UC1);
    mask.setTo(255, true); // phase == phase

    // 初始化三通道颜色图
    cv::Mat color1, color2, color3;
    color1 = cv::Mat::zeros(temp.size(), temp.type());
    color2 = cv::Mat::zeros(temp.size(), temp.type());
    color3 = cv::Mat::zeros(temp.size(), temp.type());
    int row = phase.rows;
    int col = phase.cols;

    // 基于灰度图的灰度层级，给其上色，最底的灰度值0为蓝色（255，0,0），最高的灰度值255为红色（0,0,255），中间的灰度值127为绿色（0,255,0）
    // 不要惊讶蓝色为什么是（255,0,0），因为OpenCV中是BGR而不是RGB
    for (int i = 0; i < row; ++i)
    {
        auto *c1 = color1.ptr<uchar>(i);
        auto *c2 = color2.ptr<uchar>(i);
        auto *c3 = color3.ptr<uchar>(i);
        auto *r = temp.ptr<uchar>(i);
        auto *m = mask.ptr<uchar>(i);
        for (int j = 0; j < col; ++j)
        {
            if (m[j] == 255)
            {
                if (r[j] > (3 * 255 / 4) && r[j] <= 255)
                {
                    c1[j] = 255;
                    c2[j] = 4 * (255 - r[j]);
                    c3[j] = 0;
                }
                else if (r[j] <= (3 * 255 / 4) && r[j] > (255 / 2))
                {
                    c1[j] = 255 - 4 * (3 * 255 / 4 - r[j]);
                    c2[j] = 255;
                    c3[j] = 0;
                }
                else if (r[j] <= (255 / 2) && r[j] > (255 / 4))
                {
                    c1[j] = 0;
                    c2[j] = 255;
                    c3[j] = 4 * (255 / 2 - r[j]);
                }
                else if (r[j] <= (255 / 4) && r[j] >= 0)
                {
                    c1[j] = 0;
                    c2[j] = 255 - 4 * (255 / 4 - r[j]);
                    c3[j] = 255;
                }
                else {
                    c1[j] = 0;
                    c2[j] = 0;
                    c3[j] = 0;
                }
            }
        }
    }

    // 三通道合并，得到颜色图
    vector<cv::Mat> images;
    images.push_back(color3);
    images.push_back(color2);
    images.push_back(color1);
    cv::merge(images, result);

    return result;
}

void Pdense::pmvs2() {
    GenerateOption();
    cout << root_path_<< endl;
    cout << option_name_ << endl;

    PMVS3::Soption option;
    option.init(root_path_, option_name_); // 初始化参数

    PMVS3::CfindMatch findMatch;
    findMatch.init(option); // 特征点提取在这里执行
    findMatch.run(); // 完成种子扩散与剔除

    bool bExportPLY = true;
    bool bExportPatch = false;
    bool bExportPSet = false;

//    for (int i=3; i < argc; ++i)
//    {
//        std::string option(argv[i]);
//        if (option == "PATCH")
//            bExportPatch = true;
//        if (option == "PSET")
//            bExportPSet = true;
//    }

    findMatch.write(root_path_ + "models/" + option_name_, bExportPLY, bExportPatch, bExportPSet);
    if (isPCD_)
        PLY2PCD();
    if (isGlobal_)
        Global();
}

void Pdense::GenerateOption() {
    cv::FileStorage fsSettings(config_path_, cv::FileStorage::READ);
    if(!fsSettings.isOpened()) {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    option_name_ += "-" + to_string(0) + "-" + to_string(num_image_ - 1);
    ofstream foutC;
    foutC.open(root_path_ + option_name_, std::ios::out);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
    if (!foutC.is_open())
        cerr << "Projection storage failed for picture number." << endl;

    float i;
    foutC << "# generated by genOption. mode 1. cluster: 0" << endl;
    fsSettings["option"]["level"] >> i; foutC << "level " << i << endl;
    fsSettings["option"]["csize"] >> i; foutC << "csize " << i << endl;
    fsSettings["option"]["threshold"] >> i; foutC << "threshold " << i << endl;
    fsSettings["option"]["wsize"] >> i; foutC << "wsize " << i << endl;
    fsSettings["option"]["minImageNum"] >> i; foutC << "minImageNum " << i << endl;
    fsSettings["option"]["CPU"] >> i; foutC << "CPU " << i << endl;
    fsSettings["option"]["setEdge"] >> i; foutC << "setEdge " << i << endl;
    fsSettings["option"]["useBound"] >> i; foutC << "useBound " << i << endl;
    fsSettings["option"]["useVisData"] >> i; foutC << "useVisData " << i << endl;
    fsSettings["option"]["sequence"] >> i; foutC << "sequence " << i << endl;
    fsSettings["option"]["maxAngle"] >> i; foutC << "maxAngle " << i << endl;
    fsSettings["option"]["quad"] >> i; foutC << "quad " << i << endl;
    foutC << "timages " << to_string(-1) << " " << to_string(0) << " " << to_string(num_image_ - 1) << endl;
    foutC << "oimages " << to_string(0);
    foutC.close();
}

void Pdense::PLY2PCD() const {
    pcl::PCLPointCloud2 point_cloud2;
    pcl::PLYReader reader;
    reader.read(root_path_ + "models/" + option_name_ + ".ply", point_cloud2);
    pcl::PointCloud<pcl::PointXYZ> point_cloud;
    pcl::fromPCLPointCloud2(point_cloud2, point_cloud);
    pcl::PCDWriter writer;
    writer.writeASCII(root_path_ + "models/" + option_name_ + ".pcd", point_cloud);
}

void Pdense::Combine() {
    pcl::PointCloud<pcl::PointXYZRGB> cloud_a;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_b;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_c;
    pcl::io::loadPCDFile("1-0-49.pcd", cloud_a);
    pcl::io::loadPCDFile("2-0-49.pcd", cloud_b);

    cloud_c = cloud_a + cloud_b;

    pcl::io::savePCDFileBinary("c.pcd", cloud_c);
}

//void Pdense::SetGlobal(const Estimator& estimator, const double& latitude, const double& longitude, const double& altitude) {
//    if (initGPS_)
//        return;
//    origin_T_ = estimator.Ps[WINDOW_SIZE] + estimator.Rs[WINDOW_SIZE] * estimator.tic[0];
//    geoConverter.Reset(latitude, longitude, altitude);
////    latitude_ = latitude;
////    longitude_ = longitude;
////    altitude_ = altitude;
//    initGPS_ = true;
//}

void Pdense::InitGPS(const Eigen::Vector3d& T, const double& latitude, const double& longitude, const double& altitude) {
    if (initGPS_)
        return;
    origin_T_ = T;
    geoConverter.Reset(latitude, longitude, altitude);
    initGPS_ = true;
    cerr.precision(10);
    cerr << "InitGPS successfully. " << latitude << " " << longitude << " " << altitude << endl;
}


void Pdense::Global() const {
    if (!initGPS_)
        return;

    ofstream foutC;
    foutC.open(root_path_ + "models/" + "lla" + ".txt", std::ios::out);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
    if (!foutC.is_open()) {
        cerr << "Failed to open lla.txt." << endl;
        return;
    }

    pcl::PointCloud<pcl::PointXYZRGB> pcd;
    pcl::io::loadPCDFile(root_path_ + "models/" + option_name_ + ".pcd", pcd);

    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(6);
    double lla[3];
    for (auto point : pcd) {
        geoConverter.Reverse(point.x - origin_T_.x(), point.y - origin_T_.y(), point.z - origin_T_.z(), lla[0], lla[1], lla[2]);
        foutC.precision(16);
        foutC << lla[0] << " " << lla[1] << " " << lla[2] << endl;
    }

    foutC.close();
}




