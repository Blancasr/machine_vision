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
cv::Mat IFT(cv::Mat spec);
cv::Mat LowpassFilter(cv::Mat spec, int rad);
cv::Mat Histogram(cv::Mat gray_img, cv::Scalar color, cv::Mat *hist);
cv::Mat contract(cv::Mat img, int min, int max);
cv::Mat expand(cv::Mat img, float min, float max);
cv::Mat substract_imgs(cv::Mat img1, cv::Mat img2);

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

int option_min, option_max = 30;
const int max_transformation_options = 255;
cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image;
  cv::Mat spec;
  cv::Mat spec_filtered;
  cv::Mat low_pass;
  cv::Mat gray_img;
  out_image = in_image;

  double comparison;
  int CONVOLUTION = 0;

  // Processing
  cv::createTrackbar( "Shrink value min: 0-255 ", "out_image", &option_min, max_transformation_options,0); 
  cv::createTrackbar( "Shrink value max: 0-255 ", "out_image", &option_max, max_transformation_options,0); 

  // original gray image histogram
  cv::cvtColor(in_image, gray_img, cv::COLOR_BGR2GRAY);
  cv::Mat original_hist;
  cv::Mat original_hist_img = Histogram(gray_img, cv::Scalar(0,0,255), &original_hist);

  // low-pass filter
  spec = DFT(in_image);
  spec_filtered = LowpassFilter(spec,50);
  low_pass = IFT(spec_filtered);
  
  // ---------------------------------------------------------------------
  // contract low filtered image and make histogram
  cv::Mat low_contracted = contract(low_pass, option_min, option_max);

  cv::Mat hist1;
  cv::Mat low_hist = Histogram(low_contracted, cv::Scalar(255,0,0), &hist1);
  comparison = cv::compareHist(original_hist, hist1, CONVOLUTION);
  // join histograms and show comparison
  cv::Mat img1;
  cv::bitwise_or(low_hist, original_hist_img, img1);
  cv::putText(img1, std::to_string(comparison), cv::Point(img1.rows-20, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

  // ---------------------------------------------------------------------

  // substract pixel by pixel low_contracted image to original gray image and make histogram
  cv::Mat substracted = substract_imgs(gray_img, low_contracted);

  cv::Mat hist2;
  cv::Mat subs_hist = Histogram(substracted, cv::Scalar(255, 0, 0), &hist2);
  comparison = cv::compareHist(original_hist, hist2, CONVOLUTION);

  // join histograms and show comparison
  cv::Mat img2;
  cv::bitwise_or(subs_hist, original_hist_img, img2);
  cv::putText(img2, std::to_string(comparison), cv::Point(img2.rows-20, 15),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

  // ---------------------------------------------------------------------
  
  // expand substraced image and make histogram
  cv::Mat expanded = expand(substracted, 0.0, 255.0);

  cv::Mat hist3;
  cv::Mat exp_hist = Histogram(expanded, cv::Scalar(255, 0, 0), &hist3);
  comparison = cv::compareHist(original_hist, hist3, CONVOLUTION);

  // join histograms and show comparison
  cv::Mat img3;
  cv::bitwise_or(exp_hist, original_hist_img, img3);
  cv::putText(img3, std::to_string(comparison), cv::Point(img3.rows-20, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
  
  // ---------------------------------------------------------------------

  // equalize image and show in out_image
  cv::Mat hist4, equalized_img;
  cv::equalizeHist(expanded, equalized_img);
  
  cv::Mat equalized = Histogram(equalized_img, cv::Scalar(255, 0, 0), &hist4);
  comparison = cv::compareHist(original_hist, hist4, CONVOLUTION);

  cv::Mat img4;
  cv::bitwise_or(equalized, original_hist_img, img4);
  cv::putText(img4, std::to_string(comparison), cv::Point(img4.rows-20, 15),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

  // ---------------------------------------------------------------------

  // Show image in a different window
  cv::imshow("out_image",equalized_img);
  cv::imshow("contraction",img1);
  cv::imshow("substraction",img2);
  cv::imshow("expansion",img3);
  cv::imshow("equalization",img4);

  cv::waitKey(3);

  // You must to return a 3-channels image to show it in ROS, so do it with 1-channel images
  return out_image;
}

cv::Mat substract_imgs(cv::Mat img1, cv::Mat img2) {

  cv::Mat result =  img1.clone();

  if (img1.rows != img2.rows || img1.cols != img2.cols) {
    std::cerr << "matrix sizes do not match" << std::endl;
    return result;
  }

  for ( int i = 0; i < result.rows ; i++) {
    for ( int j = 0; j < result.cols ; j++) {
      result.at<uchar>(i,j) = img1.at<uchar>(i,j) - img2.at<uchar>(i,j);
    }
  }
  return result;
}

cv::Mat expand(cv::Mat img, float min, float max) {
  
  float img_max = -1, img_min = 1000;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if (img.at<float>(i,j) > img_max) {
        img_max = img.at<float>(i,j);
      }
      if (img.at<float>(i,j) < img_min) {
        img_min = img.at<float>(i,j);
      }
    }
  }
  
  img_max = option_max;
  img_min = option_min;
  /*
  min = 0;
  max = 255;
  */

  cv::Mat expanded = img.clone();
  for (int i = 0; i < expanded.rows; i++) {
    for (int j = 0; j < expanded.cols; j++) {
      expanded.at<uchar>(i,j) = ((img.at<uchar>(i,j) - img_min) / (img_max - img_min) ) * (max - min) + min;
    }
  }
  return expanded;
}

cv::Mat contract(cv::Mat img, int min, int max) {
  cv::Mat contracted = img.clone();

  float img_max = -1, img_min = 1000;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if (img.at<float>(i,j) > img_max) {
        img_max = img.at<float>(i,j);
      }
      if (img.at<float>(i,j) < img_min) {
        img_min = img.at<float>(i,j);
      }
    }
  }

  if (min > max) {
      min = 50;
      max = 150;
  }
  for (int i = 0; i < contracted.rows; i++) {
    for (int j = 0; j < contracted.cols; j++) {
      contracted.at<float>(i,j) = ( (max - min) / (img_max - img_min) ) * (img.at<float>(i,j) - img_min) + min;
    }
  }

  return contracted;
}

cv::Mat Histogram(cv::Mat gray_img, cv::Scalar color, cv::Mat *hist) {
  
  int histSize = 256;

  float range[] = { 0, 256 };
  const float* histRange = { range };
  bool uniform = true, accumulate = false;

  // calculate histogram
  cv::calcHist( &gray_img, 1, 0, cv::Mat(), *hist, 1, &histSize, &histRange, uniform, accumulate );

  // draw histogram
  int hist_w = 512, hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );
    
  cv::Mat histImage( hist_h, hist_w, CV_32F, cv::Scalar(0,0,0));
  normalize(*hist, *hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  
  // Draw the intensity line for histograms
  for( int i = 1; i < histSize; i++ ) {
    line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist->at<float>(i-1)) ),
      cv::Point( bin_w*(i), hist_h - cvRound(hist->at<float>(i)) ),
      cv::Scalar( color[0], color[1], color[2]), 2, 8, 0 );   
  }

  return histImage;
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

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}