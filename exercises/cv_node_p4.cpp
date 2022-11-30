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
void HoughCircles(cv::Mat gray, cv::Mat* out_img, int max_rad);
void HoughLines(cv::Mat edges, cv::Mat* rgb_im, int line_thresh);
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

int canny_thresh , option  , max_option = 3, line_accumulator, hough_max_rad, aspect_contour_ratio, pixels_contour;
const int max_transformation_options = 255;
cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image;
  out_image = in_image;

  cv::createTrackbar( "0: Hough 1:Contours 2:Centroids 3:Ball ", "out_image", &option, max_option,0); 
  cv::createTrackbar( "Canny thresh [0-255] ", "out_image", &canny_thresh, 255,0); 
  cv::createTrackbar( "Hough lines accumulator [0-300] ", "out_image", &line_accumulator, 300,0);
  cv::createTrackbar( "Hough radius value max[0-200] ", "out_image", &hough_max_rad, 200,0);
  cv::createTrackbar( "Aspect ratio value contours * 0.1 [0-4]", "out_image", &aspect_contour_ratio, 4,0);
  cv::createTrackbar( "Number of pixels contours [0-1000]", "out_image", &pixels_contour, 1000,0);

  cv::Mat edges, gray, gaussian_edges, gauss, hsv, mask, aux = in_image.clone();;

  std::vector<std::vector<cv::Point> > contours, all_contours;
  std::vector<cv::Vec4i> hierarchy, all_hierarchy ;
  cv::Rect rect;
  float aspectRatio;
  double area;

  std::vector<cv::Rect> bounding_boxes;
  cv::Moments moment;
  double cx, cy;


  bool show_all = false; // turn to true to view auxiliar images

  switch(option) {
    case 0:   // HOUGH

      cv::cvtColor(out_image, gray, cv::COLOR_BGR2GRAY);
      Canny(gray, edges, canny_thresh, canny_thresh*2);
      HoughLines(edges, &out_image, line_accumulator);
      HoughCircles(gray, &out_image, hough_max_rad);
      break;

    case 1:  // CONTOURS:

      cv::cvtColor(out_image, gray, cv::COLOR_BGR2GRAY);
      GaussianBlur(gray, gauss, cv::Size(5,5), 0);
      Canny(gauss, gaussian_edges, canny_thresh, canny_thresh*2);

      cv::findContours( gaussian_edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
 
      for ( int i = 0; i < (int)contours.size() ; i++) {
        
        area = cv::contourArea(contours[i]);
    
        // aspect ratio
        rect = cv::boundingRect(contours[i]);
        aspectRatio = rect.width / rect.height;

        // area of the contour must be higher than slider and difference between 1 and aspectRatio hugher than 0.1
        if( area >= pixels_contour && (std::abs(1 - aspectRatio) < 0.1 * aspect_contour_ratio) ) {
          cv::drawContours( out_image, contours, i, cv::Scalar(0,0,255), 2, cv::LINE_8, hierarchy, 1 );
        }
      }
     
      if ( show_all ) {
        // view all contours
        cv::findContours( gaussian_edges, all_contours, all_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
        cv::drawContours( aux, all_contours, -1, cv::Scalar(0,0,255), 2, cv::LINE_8, all_hierarchy, 1 );
        cv::imshow("aux", aux);
      }
      
      break;

    case 2:   // CENTROIDS AND BOUNDING BOXES
      
      // gray image blur and canny
      cv::cvtColor(out_image, gray, cv::COLOR_BGR2GRAY);
      GaussianBlur(gray, gauss, cv::Size(5,5), 0);
      Canny(gauss, gaussian_edges, canny_thresh, canny_thresh*2);

      cv::findContours( gaussian_edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

      for (int i = 0; i < (int)contours.size(); i++) {
        
        area = cv::contourArea(contours[i]);

        // centroids:
        moment = cv::moments( contours[i] );
        cx = moment.m10 / moment.m00;
        cy = moment.m01 / moment.m00;

        // bounding boxes
        bounding_boxes.push_back( cv::boundingRect( contours[i] ) );
        aspectRatio = bounding_boxes[i].width / bounding_boxes[i].height;

        // draw in out_image if slider values fit with contour
        if( area >= pixels_contour && (std::abs(1 - aspectRatio) < 0.1 * aspect_contour_ratio) ) {
          cv::rectangle( out_image, bounding_boxes[i].tl() , bounding_boxes[i].br(), cv::Scalar(0,255,0), 2 );
          cv::circle( out_image, cv::Point(cx,cy), 0, random_color(), 4 );

          cv::drawContours( aux, contours, i, cv::Scalar(0,0,255), 2, cv::LINE_8, all_hierarchy, 1 );
        }
      }

      if (show_all) {
        cv::imshow("aux",aux);
      }
      break;

    case 3:   // BALL

      // hsv and gray images
      cv::cvtColor(out_image, hsv, cv::COLOR_BGR2HSV);
      cv::cvtColor(out_image, gray, cv::COLOR_BGR2GRAY);

      // capture all blue objects and make a gray scale image of them
      cv::inRange(hsv, cv::Scalar(85,85,20),cv::Scalar(100,255,255), mask );
      cv::bitwise_and( gray, mask, aux);

      // capture blue circles from blue objects gray scale image
      HoughCircles(aux, &out_image, hough_max_rad);

      if (show_all) {
        cv::imshow("aux",aux);
      }
      
      break;
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

void HoughCircles(cv::Mat gray, cv::Mat* out_img, int max_rad) {

    cv::medianBlur(gray, gray, 5);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                 gray.rows/16,  // change this value to detect circles with different distances to each other
                 100, 30, 0, max_rad);
    
    for( size_t i = 0; i < circles.size(); i++ ) {
      
      cv::Vec3i c = circles[i];
      cv::Point center = cv::Point(c[0], c[1]);

      // circle center
      cv::circle( *out_img, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
      // circle outline
      int radius = c[2];
      cv::circle( *out_img, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);      
    }
}

void HoughLines(cv::Mat edges, cv::Mat* rgb_im, int line_thresh){
      // option 1 - Standard
  
      // Standard Hough Line Transform
      std::vector<cv::Vec2f> lines; // will hold the results of the detection (rho, theta)
      cv::HoughLines( edges, lines, 1, CV_PI/180, line_thresh, 0, 0 ); // runs the actual detection

      // Draw the lines
      for( size_t i = 0; i < lines.size(); i++ ) {
          float rho = lines[i][0], theta = lines[i][1];
          cv::Point pt1, pt2;
          double a = cos(theta), b = sin(theta);
          double x0 = a*rho, y0 = b*rho;
          pt1.x = cvRound(x0 + 1000*(-b));
          pt1.y = cvRound(y0 + 1000*( a));
          pt2.x = cvRound(x0 - 1000*(-b));
          pt2.y = cvRound(y0 - 1000*( a));
          line( *rgb_im, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA );
      }
      /*
      // option 2 - Probabilistic

      // Probabilistic Line Transform
      std::vector<cv::Vec4i> linesP; // will hold the results of the detection
      cv::HoughLinesP( edges , linesP, 1, CV_PI/180, line_thresh, 50, 10 ); // runs the actual detection
      
      // Draw the lines
      for( size_t i = 0; i < linesP.size(); i++ ) {
          cv::Vec4i l = linesP[i];
          line( *rgb_im, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA );
      }
      */
}

int main(int argc, char * argv[])
{
  srand(time(NULL));
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}