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
void convert_CMY(cv::Mat in_img, cv::Mat* out_img);
void convert_HSI(cv::Mat in_img, cv::Mat* out_img);
void convert_HSV(cv::Mat in_img, cv::Mat* out_img);
void HSV_to_HSI(cv::Mat bgr_img, cv::Mat *hsv_img);

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

    //friend cv::Mat image_processing(const cv::Mat in_image);

  private:

    //int color_space;
    //const int max_color_space = 5;

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

/**
  TO-DO
*/
int color_space;
const int max_color_space = 5;
cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image;
  out_image = in_image;
  
  // Processing
  cv::createTrackbar( "Element:\n 0: RGB \n 1: CMY \n 2: HSI \n 3: HSV \n 4: HSVOpenCV \n 5: HSI OpenCV ", "out_image", &color_space, max_color_space,0 ); 

  switch(color_space){
    case 0: 
      out_image = in_image; break;
    case 1: 
      convert_CMY(in_image,&out_image); break;
    case 2: 
      convert_HSI(in_image,&out_image); break;
    case 3: 
      convert_HSV(in_image,&out_image); break;
    case 4: 
      cvtColor(in_image, out_image, cv::COLOR_BGR2HSV); break;
    case 5: 
      cvtColor(in_image, out_image, cv::COLOR_BGR2HSV);
      HSV_to_HSI(in_image, &out_image);
      break;
  }

  // Show image in a different window
  cv::imshow("out_image",out_image);
  cv::waitKey(3);

  // You must to return a 3-channels image to show it in ROS, so do it with 1-channel images
  //cv::cvtColor(out_image, out_image, cv::COLOR_GRAY2BGR);
  return out_image;
}

void convert_CMY(cv::Mat in_img, cv::Mat* out_img){

  for (int i = 0; i < in_img.rows ; i++){
    for ( int j = 0; j < in_img.cols ; j++){
      out_img->at<cv::Vec3b>(i,j)[0] = 255 - (uint)in_img.at<cv::Vec3b>(i,j)[0];
      out_img->at<cv::Vec3b>(i,j)[1] = 255 - (uint)in_img.at<cv::Vec3b>(i,j)[1];
      out_img->at<cv::Vec3b>(i,j)[2] = 255 - (uint)in_img.at<cv::Vec3b>(i,j)[2];
    }
  }
}

void convert_HSI(cv::Mat in_img, cv::Mat* out_img){
  float R,G,B,H,S,I,min,den;

  for (int i = 0; i < in_img.rows ; i++){
    for ( int j = 0; j < in_img.cols ; j++){
      //normalize
      B = (uint)in_img.at<cv::Vec3b>(i,j)[0]/255.0;
      G = (uint)in_img.at<cv::Vec3b>(i,j)[1]/255.0;
      R = (uint)in_img.at<cv::Vec3b>(i,j)[2]/255.0;
      
      min = R < G ? R : G;
      min = min < B ? min : B;

      //transformation of values from RGB to HSI
      den = std::pow( std::pow((R-B),2) + (R-B)*(G-B), 0.5);
      if(den != 0.0){
        H = std::acos((((R-G)+(R-B))/2) / den);
      }else{
        H = std::acos(1);
      }
      S = 1 - 3*min/(R+G+B);
      I = (R+G+B)/3;
      
      if(B > G){
        H = 360 - H;
      }

      //establish new color
      out_img->at<cv::Vec3b>(i,j)[0] = (uint)((H/360)*255);
      out_img->at<cv::Vec3b>(i,j)[1] = (uint)(S*255);
      out_img->at<cv::Vec3b>(i,j)[2] = (uint)(I*255);
    }
  }
}

void convert_HSV(cv::Mat in_img, cv::Mat* out_img){
  float R,G,B,H,S,V, min, max,den;

  for (int i = 0; i < in_img.rows ; i++){
    for ( int j = 0; j < in_img.cols ; j++){
      //normalize
      B = (uint)in_img.at<cv::Vec3b>(i,j)[0]/255.0;
      G = (uint)in_img.at<cv::Vec3b>(i,j)[1]/255.0;
      R = (uint)in_img.at<cv::Vec3b>(i,j)[2]/255.0;

      min = R < G ? R : G;
      min = min < B ? min : B;

      max = R > G ? R : G;
      max = max > B ? max : B;

      //transformation of values from RGB to HSV
      den = std::pow( std::pow((R-B),2) + (R-B)*(G-B), 0.5);
      if(den != 0.0){
         H = std::acos((((R-G)+(R-B))/2) / den);
      }else{
        H = std::acos(1);
      }
      S = 1 - 3*min/(R+G+B);
      V = max;
      
      if(B > G){
        H = 360.0 - H;
      }
      //establish new color
      out_img->at<cv::Vec3b>(i,j)[0] = (H/360)*255;
      out_img->at<cv::Vec3b>(i,j)[1] = S*255;
      out_img->at<cv::Vec3b>(i,j)[2] = V*255;
    }
  }
}

void HSV_to_HSI(cv::Mat bgr_img, cv::Mat *hsv_img){
  double I, B,G,R;

  for (int i = 0; i < bgr_img.rows ; i++){
    for ( int j = 0; j < bgr_img.cols ; j++){
      B = (uint)bgr_img.at<cv::Vec3b>(i,j)[0]/255.0;
      G = (uint)bgr_img.at<cv::Vec3b>(i,j)[1]/255.0;
      R = (uint)bgr_img.at<cv::Vec3b>(i,j)[2]/255.0;

      I = (R+G+B)/3;
      hsv_img->at<cv::Vec3b>(i,j)[2] = (uint)(I*255);
    }
  }
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}
