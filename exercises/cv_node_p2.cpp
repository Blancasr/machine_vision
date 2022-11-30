#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <image_transport/image_transport.hpp>

#include <memory>
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <cmath>


cv::Mat image_processing(const cv::Mat in_image);
cv::Mat DFT(cv::Mat in_img);
cv::Mat fftShift(cv::Mat magI);
cv::Mat visual_spectrum(const cv::Mat &complexI);
cv::Mat IFT(cv::Mat spec);
cv::Mat HighpassFilter(cv::Mat spec,int rad);
cv::Mat LowpassFilter(cv::Mat spec, int rad);
cv::Mat threshold_operation(cv::Mat src, float threshold);

class ComputerVisionSubscriber : public rclcpp::Node
{
  public:
    ComputerVisionSubscriber()
    : Node("opencv_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
   

      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/rgb/image_raw", qos, std::bind(&ComputerVisionSubscriber::topic_callback, this, std::placeholders::_1));
    
      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "cv_image", qos);
    }

  private:

    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw =  cv_ptr->image;

      // Image processing
      cv::Mat cv_image = image_processing(image_raw);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; // >> message to be sent
      img_bridge.toImageMsg(out_image); // from cv_bridge to sensor_msgs::Image

      // Publish the data
      publisher_ -> publish(out_image);

    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
   
};

int option;
const int max_transformation_options = 4;
cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image;
  out_image = in_image;

  cv::Mat spec;
  cv::Mat spec_filtered;
  cv::Mat img1;
  cv::Mat img2;
  
  // Processing
  cv::createTrackbar( "Element:\n 0: Original \n 1: Fourier \n 2: Highpass Filter \n 3: Lowpass Filter \n 4: AND ", "out_image", &option, max_transformation_options,0 ); 

  switch(option){
    case 0: 
      cv::cvtColor(in_image, out_image, cv::COLOR_BGR2GRAY);
      break;
    case 1: 
      spec = DFT(in_image);
      out_image = visual_spectrum(spec);
      break;
    case 2: 
      spec = DFT(in_image);
      spec_filtered = HighpassFilter(spec,50);
      out_image = IFT(spec_filtered);
      //out_image = visual_spectrum(spec_filtered);
      break;
    case 3: 
      spec = DFT(in_image);
      spec_filtered = LowpassFilter(spec,50);
      out_image = IFT(spec_filtered);
      //out_image = visual_spectrum(spec_filtered);
      break;
    case 4: 
      spec = DFT(in_image);
      spec_filtered = HighpassFilter(spec,50);
      img1 = IFT(spec_filtered);
      img1 = threshold_operation(img1, 0.4);

      spec = DFT(in_image);
      spec_filtered = LowpassFilter(spec,50);
      img2 = IFT(spec_filtered);
      img2 = threshold_operation(img2, 0.6);

      cv::bitwise_and(img1, img2, out_image);
      //out_image = img2;
      //out_image = img1;
      break;
  }

  // Show image in a different window
  cv::imshow("out_image",out_image);
  cv::waitKey(3);

  // You must to return a 3-channels image to show it in ROS, so do it with 1-channel images
  return out_image;
}

cv::Mat fftShift(cv::Mat magI) {
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    cv::Mat rearranged = magI.clone();
    
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    cv::Mat q0(rearranged, cv::Rect(0, 0, cx, cy));   // Top-Left 
    cv::Mat q1(rearranged, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(rearranged, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(rearranged, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    return rearranged;
}

cv::Mat visual_spectrum(const cv::Mat &complexI) {

    cv::Mat complexImg = fftShift(complexI);

    // Transform the real and complex values to magnitude
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    cv::Mat planes_spectrum[2];
    split(complexImg, planes_spectrum);       // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes_spectrum[0], planes_spectrum[1], planes_spectrum[0]);// planes[0] = magnitude
    cv::Mat spectrum = planes_spectrum[0];

    // Switch to a logarithmic scale
    spectrum += cv::Scalar::all(1);
    cv::log(spectrum, spectrum);

    // Normalize
    cv::normalize(spectrum, spectrum, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
                                                      // viewable image form (float between values 0 and 1).
    return spectrum;
}

cv::Mat DFT(cv::Mat in_img) {

    cv::cvtColor(in_img, in_img, cv::COLOR_BGR2GRAY);

    // Expand the image to an optimal size ( rendimiento del algoritmo).     
    cv::Mat padded;  
    int m = cv::getOptimalDFTSize( in_img.rows );
    int n = cv::getOptimalDFTSize( in_img.cols ); // on the border add zero values
    cv::copyMakeBorder(in_img, padded, 0, m - in_img.rows, 0, n - in_img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Make place for both the complex and the real values
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    // Make the Discrete Fourier Transform
    cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT);      // this way the result may fit in the source matrix

    return complexI; 
}

cv::Mat IFT(cv::Mat spec) {
    cv::Mat inverseTransform;
    idft(spec, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);
    return inverseTransform;
}

cv::Mat LowpassFilter(cv::Mat spec, int rad) {

  // create the image that corresponds to the filter
  cv::Mat filtered_spec, two_planes_filter;
  cv::Mat filter = cv::Mat::zeros(spec.rows, spec.cols, CV_32F);
  std::vector<cv::Mat> planes;
  
  cv::Point center(filter.cols/2 , filter.rows/2);
  cv::Scalar color(255,255,255);
  cv::circle(filter,center,rad,color,cv::FILLED);

  // create the a two-dimensional image of the filter
  planes.push_back(filter);
  planes.push_back(filter);
  merge(planes, two_planes_filter);

  // multiply both spectrums : our complex image with the two-dimensional filter
  two_planes_filter = fftShift(two_planes_filter);
  mulSpectrums(two_planes_filter, spec, filtered_spec, 0);

  return filtered_spec;
}

cv::Mat HighpassFilter(cv::Mat spec, int rad) {

  //create the image that corresponds to the filter
  cv::Mat filtered_spec, two_planes_filter;
  cv::Mat filter = cv::Mat::ones(spec.rows, spec.cols, CV_32F);
  std::vector<cv::Mat> planes;
  
  cv::Point center(filter.cols/2 , filter.rows/2);
  cv::Scalar color(0,0,0);
  cv::circle(filter,center,rad,color,cv::FILLED);

  // create the a two-dimensional image of the filter
  planes.push_back(filter);
  planes.push_back(filter);
  merge(planes, two_planes_filter);

  // multiply both spectrums : our complex image with the two-dimensional filter
  two_planes_filter = fftShift(two_planes_filter);
  mulSpectrums(two_planes_filter, spec, filtered_spec, 0);

  return filtered_spec;
}

cv::Mat threshold_operation(cv::Mat src, float threshold) {
  cv::Mat out = src.clone();

  // analize each pixel and change its value according to the threshold
  for ( int i = 0; i < out.rows; i++) {
    for (int j = 0; j < out.cols ; j++) {
      float value = (float)src.at<float>(i,j);
      if(value > threshold){
        out.at<float>(i,j) = (float)255.0;
      } else {
        out.at<float>(i,j) = (float)0.0;
      }
    }
  }
  return out;
}


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}