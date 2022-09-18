//
// Created by hzx on 2022/7/11.
//

#ifndef SRC_PDENSE_H
#define SRC_PDENSE_H

#include <iostream>
#include <string>
#include "../estimator/estimator.h"
#include "base/pmvs/findMatch.h"
#include "base/pmvs/option.h"
#include "../../../global_fusion/ThirdParty/GeographicLib/include/LocalCartesian.hpp"

class Pdense {
public:
    Pdense() = default;
    explicit Pdense(const string& config_path);
    Pdense(const string& config_path, const string& option_name);

    string root_path_;
    string config_path_;
    string option_name_ = "option-0000";
    int max_num_ = 0;
    int num_image_ = 0;
    GeographicLib::LocalCartesian geoConverter;
    Eigen::Vector3d origin_T_;

    void write(const Estimator& estimator, cv::Mat img);

    void pmvs2() const;

    static cv::Mat Gray2Color(cv::Mat &phase);

    void PLY2PCD() const;

    void generateOption();

    static void Combine() ;

    void SetGlobal(const Estimator& estimator, const double& latitude, const double& longitude, const double& altitude);
    bool initGPS_ = false;
//    double latitude_;
//    double longitude_;
//    double altitude_;
    void Global() const;

private:
};

#endif //SRC_PDENSE_H
