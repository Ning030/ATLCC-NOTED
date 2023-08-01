/**
 * @file    HistogramHandler.ipp
 *
 * @author  btran
 *
 */

namespace perception
{
template <typename PointCloudType>
bool HistogramHandler::update(const std::vector<cv::Mat>& grayImgs,
                              const std::vector<typename pcl::PointCloud<PointCloudType>::Ptr>& inClouds,
                              const CameraInfo& cameraInfo, const Eigen::Affine3d& affine)
{
    //再次判断是否相等
    if (grayImgs.size() != inClouds.size()) {
        DEBUG_LOG("mismatch number of images and point clouds");
        return false;
    }

    //再次判断是否为空
    if (grayImgs.empty()) {
        DEBUG_LOG("empty data");
        return false;
    }

    //循环update
    for (std::size_t i = 0; i < grayImgs.size(); ++i) {
        const auto& inCloud = inClouds[i];
        const auto& grayImg = grayImgs[i];

        this->update<PointCloudType>(grayImg, inCloud, cameraInfo, affine);
    }

    return true;
}
/***
* @brief  处理单帧图像，更新直方图
* @param  grayImg - 灰度图
* @param  inCloud - 点云
* @param  cameraInfo - 相机内参类
* @param  affine - 当前lidar-to-camera变换
* @return bool
*/
template <typename PointCloudType>
bool HistogramHandler::update(const cv::Mat& grayImg, const typename pcl::PointCloud<PointCloudType>::Ptr& inCloud,
                              const CameraInfo& cameraInfo, const Eigen::Affine3d& affine)
{
    //判断类别，是否为灰度图
    if (grayImg.type() != CV_8UC1) {
        DEBUG_LOG("need to use gray image");
        return false;
    }

    //根据变换将点云变换到新的坐标系下
    typename pcl::PointCloud<PointCloudType>::Ptr alignedCloud(new pcl::PointCloud<PointCloudType>());
    pcl::transformPointCloud(*inCloud, *alignedCloud, affine.matrix());

    //遍历点云，
    for (const auto& point : alignedCloud->points) {
        //当前点投影到图像
        cv::Point imgPoint = projectToImagePlane(point, cameraInfo);
        if (!this->validateImagePoint(grayImg, imgPoint)) {
            continue;
        }
        // m_binFraction = MAX_BINS / m_numBins;
        // 等于1
        int intensityBin = point.intensity / m_binFraction;
        int grayBin = grayImg.ptr<uchar>(imgPoint.y)[imgPoint.x] / m_binFraction;

        m_intensityHist.at<double>(intensityBin)++;
        m_grayHist.at<double>(grayBin)++;
        m_jointHist.at<double>(grayBin, intensityBin)++;

        m_intensitySum += intensityBin;
        m_graySum += grayBin;

        m_totalPoints++;
    }

    return true;
}
}  // namespace perception
