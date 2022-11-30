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
#include <time.h>
#include <stdlib.h>


cv::Mat image_processing(const cv::Mat in_image);
cv::Scalar random_color(void);

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

int iterations , element_option  , max_distance = 3, kernel_size;
cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image;
  out_image = in_image.clone();

  cv::createTrackbar( "Iterations [0-100]", "out_image", &iterations, 100,0); 
  cv::createTrackbar( "ELement:\n0: Rect - 1: Cross - 2: Ellipse", "out_image", &element_option, 2,0); 
  cv::createTrackbar( "Kernel size:\n2n + 1 ", "out_image", &kernel_size, 5,0);
  cv::createTrackbar( "Max distance ", "out_image",&max_distance, 8,0);

  
  // INTRINSIC parameters:
  float fx = 522.1910329546544, fy = 522.1910329546544, cx = 320.5, cy =240.5;
  float K_array[3][3] = {{fx,  1, cx},
                         {0, fy, cy},
                         {0,  0, 1}};
  cv::Mat K(3, 3, CV_32F, K_array);

  // EXTRINSIC parameters:
  // translation = {{0.125}, {0}, {0.9864}}; // en gazebo
  // rotation
  float roll_angle = 0.0;
  float pitch_angle = 0.0;
  float yaw_angle = 0.0;

  float roll_array[9] = {1, 0, 0, 0, std::cos(roll_angle), -std::sin(roll_angle), 0, std::sin(roll_angle), std::cos(roll_angle)};
  float picth_array[9] = {std::cos(-pitch_angle), 0, std::sin(-pitch_angle), 0, 1, 0, -std::sin(-pitch_angle), 0, std::cos(-pitch_angle)};
  float yaw_array[9] = {std::cos(-yaw_angle), -std::sin(-yaw_angle),0, std::sin(-yaw_angle), std::cos(-yaw_angle),0, 0, 0, 1};
  
  cv::Mat roll(3, 3, CV_32F, roll_array);
  cv::Mat pitch(3, 3, CV_32F, picth_array);
  cv::Mat yaw(3, 3, CV_32F, yaw_array);
  cv::Mat R = yaw * pitch * roll;

  float RT_array[12] = {R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2), 0,
                          R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2), 0.9864,
                          R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2), 0.125 };
  cv::Mat RT(3, 4, CV_32F, RT_array);

  float x[2], y[2];
  float side ;
  int b = 0, g = 0, r= 255;
  cv::Scalar color(b,g,r);

  if (max_distance >= 3) {
    for (int j = 3; j<= max_distance; j++) {
      side = 1;
      for (int i = 0; i<2; i++) {
        float pos_array[4] = {side, 0, (float)j, 1}; // coordenada 3d en el espacio respecto  los ejes de la imagen
        cv::Mat pos(4, 1, CV_32F, pos_array);

        cv::Mat pixel_2d = K * RT * pos;

        float s = pixel_2d.at<float>(2,0);

        x[i] = pixel_2d.at<float>(0,0) /s;
        y[i] = pixel_2d.at<float>(1,0) /s;

        // std::cout << x[i] << " " << y[i]  << std::endl;
        side = -1;
      }
      b += 15;
      g += 40;
      r -= 40;
      color = cv::Scalar(b,g,r);
      cv::line (out_image, cv::Point(x[0],y[0]), cv::Point(x[1],y[1]), color, 4, cv::LINE_8, 0);
      cv::putText(out_image, std::to_string(j), cv::Point(x[0]+10,y[0]),cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2, 4,false);
    }
  }
  

  // SKELETON
  cv::Mat hsv, wood, skeleton, open, element, temp;

    // imagen que contiene solo los objetos a max_distance
  cv::Mat img_at_dist = in_image.clone();
  for ( int i = 0; i<img_at_dist.rows; i++) {
    for (int j = 0; j<img_at_dist.cols; j++) {
      if(max_distance >= 3 && (i > y[0] || j > x[0] || j < x[1]) ) {
        img_at_dist.at<cv::Vec3b>(i,j)[0] = 0;
        img_at_dist.at<cv::Vec3b>(i,j)[1] = 0;
        img_at_dist.at<cv::Vec3b>(i,j)[2] = 0;
      }
    }
  }

  // detectar madera clara
  cv::cvtColor(img_at_dist, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv, cv::Scalar(10,120,100),cv::Scalar(27,200,255), wood );
  // cv::imshow("at distance", wood); // mostrar objetos detectados

  // crear una imgaen esqueleto vacia
  skeleton = cv::Mat::zeros(in_image.size(), CV_8UC1);

  for (int i = 0; i < iterations; i++) {
    // realizar apertura de la imagen
    element = cv::getStructuringElement( element_option, cv::Size( 2*kernel_size + 1, 2*kernel_size+1 ), cv::Point( kernel_size, kernel_size ) );
    cv::morphologyEx( wood, open, 2, element ); // operation 2 is OPEN 

    // restar imagen abierta a la original 
    temp = wood - open;

    // redefinir esqueleto uniendo el equeleto con la imagen temp
    cv::bitwise_or( skeleton, temp, skeleton);

    // erosionar imagen original
    cv::erode( wood, wood, element );
  }

  // dibujar esqueleto en out_image
  for (int i = 0; i < out_image.rows; i++) {
		for (int j = 0; j < out_image.cols; j++) {
			cv::Scalar intensity = skeleton.at<uchar>(i, j);
			if (intensity.val[0] == 255) {
				out_image.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
			}
		}
	}
  
  // Show image in a different window
  cv::imshow("out_image",out_image);

  cv::waitKey(3);

  // You must to return a 3-channels image to show it in ROS, so do it with 1-channel images
  return out_image;
}

cv::Scalar random_color(void) {
  
  float B = rand() %255;
  float G = rand() %255;
  float R = rand() %255;

  return cv::Scalar(R,G,B);
}


int main(int argc, char * argv[])
{
  srand(time(NULL));

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}