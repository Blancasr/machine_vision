/*
Autor: Blanca Soria
Partes implementadas:
- DetecciÃ³n de pelota en 2D y 3D
- Reconocimiento de mesa en 2D
- IdentificaciÃ³n de mesa y objetos con Yolo
*/

#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <pthread.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <image_transport/image_transport.hpp>

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types_conversion.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/features/normal_3d.h>


// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
std::vector<std::string> classes;
cv::dnn::dnn4_v20191202::Net net;

cv::Mat K, RT;
std::vector<std::vector<float> > ball_points;
bool diningTable;

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud);
cv::Mat image_processing(const cv::Mat in_image);
//cv
int HoughCircles(cv::Mat gray, cv::Mat* out_img);
void proyection_lines(cv::Mat *out_image, int max_distance , cv::Mat K, cv::Mat RT);
void proyect_circle(float x, float y, float z, cv::Mat* out_image, cv::Mat K, cv::Mat RT);
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& out);// Remove the bounding boxes with low confidence using non-maxima suppression
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);// Draw the predicted bounding box
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);// Get the names of the output layers

//pcl
pcl::PointCloud<pcl::PointXYZRGB> color_filter(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud, float h_min, float h_max, float s_min, float s_max, float v_min, float v_max);
pcl::PointCloud<pcl::PointXYZRGB> create_cube(float L, float x, float y, float z, float r, float g, float b );
pcl::PointCloud<pcl::PointXYZRGB> proyection_cubes( int N );
std::vector<std::vector<float> > extract_model(int model, pcl::PointCloud<pcl::PointXYZRGB> *pointcloud, bool show_model);

pthread_mutex_t mutex;

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

      diningTable = false;
      // Load names of classes
      std::string classesFile = "/root/computervision_ws/src/computer_vision/cfg/coco.names";
      ifstream ifs(classesFile.c_str());
      std::string line;
      while (getline(ifs, line)) classes.push_back(line);

      // Give the configuration and weight files for the model
      std::string modelConfiguration = "/root/computervision_ws/src/computer_vision/cfg/yolov3.cfg";
      std::string modelWeights = "/root/computervision_ws/src/computer_vision/cfg/yolov3.weights";

      // Load the network
      net = cv::dnn::dnn4_v20191202::readNetFromDarknet(modelConfiguration, modelWeights);
      net.setPreferableBackend(cv::dnn::dnn4_v20191202::DNN_TARGET_CPU);
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

class PCLSubscriber : public rclcpp::Node
{
  public:
    PCLSubscriber()
    : Node("pcl_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      subscription_3d_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/head_front_camera/depth_registered/points", qos, std::bind(&PCLSubscriber::topic_callback_3d, this, std::placeholders::_1));
    
      publisher_3d_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pcl_points", qos);
    }

  private:
    void topic_callback_3d(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {    
      // Convert to PCL data type
      pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
      pcl::fromROSMsg(*msg, point_cloud);     

      pcl::PointCloud<pcl::PointXYZRGB> pcl_pointcloud = pcl_processing(point_cloud);
      
      // Convert to ROS data type
      sensor_msgs::msg::PointCloud2 output;
      pcl::toROSMsg(pcl_pointcloud, output);
      output.header = msg->header;

      // Publish the data
      publisher_3d_ -> publish(output);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_3d_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_3d_;
};


/**
  TO-DO
*/
cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image = in_image.clone();
  
  cv::Mat gray, hsv, mask, aux = in_image.clone();

  // hsv and gray images
  cv::cvtColor(out_image, hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(out_image, gray, cv::COLOR_BGR2GRAY);

  // capture all blue objects and make a gray scale image of them
  cv::inRange(hsv, cv::Scalar(85,85,20),cv::Scalar(100,255,255), mask );
  cv::bitwise_and( gray, mask, aux);

  // capture blue circles from blue objects gray scale image
  int circles = HoughCircles(aux, &out_image);
  
  // si hay pelota azul
  if( circles > 0){

    //INTRINSIC parameters of camera:
    float fx = 522.1910329546544, fy = 522.1910329546544, cx = 320.5, cy =240.5;
    float K_array[3][3] = {{fx,  1, cx},
                           {0, fy, cy},
                           {0,  0, 1}};
    K = cv::Mat(3, 3, CV_32F, K_array);
  
    // EXTRINSIC parameters of camera:
    // translation = {{0.125}, {0}, {0.9864}}; en gazebo
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
    RT = cv::Mat(3, 4, CV_32F, RT_array);

    proyection_lines(&out_image, 8, K, RT);

    pthread_mutex_lock(&mutex);
    std::vector<std::vector<float> > center_ball = ball_points;
    pthread_mutex_unlock(&mutex);

    float RT_array2[12] = {R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2), 0,
                            R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2), -0.9864,
                            R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2), 0.125 };
    RT = cv::Mat(3, 4, CV_32F, RT_array2);

    for(int i = 0; i<(int)center_ball.size() ; i++) {
      proyect_circle(center_ball[i][0], center_ball[i][1], center_ball[i][2], &out_image, K, RT);
    }

  }else { // si no hay objetos azules
    cv::Mat blob ;
    // Create a 4D blob from a out_image.
    cv::dnn::dnn4_v20191202::blobFromImage(out_image, blob, 1/255.0, cv::Size(out_image.cols, out_image.rows), cv::Scalar(0,0,0), true, false);

    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time for a frame : %.2f ms", t);
    putText(out_image, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

    // Remove the bounding boxes with low confidence
    postprocess(out_image, outs);
    if(!diningTable) {
      out_image = in_image.clone();
      std::cout << "NO DINING TABLE " << std::endl;
    } else {
      std::cout << "DINING TABLE " << std::endl;
    }
    diningTable = false;
  }

  return out_image;
}

int HoughCircles(cv::Mat gray, cv::Mat* out_img) {

    cv::medianBlur(gray, gray, 5);

    double  inv_accumulator = 1;    // Inverse ratio of the accumulator resolution to the image resolution.
    double 	minDist = gray.rows/8; // Minimum distance between the centers of the detected circles
    double 	param1 = 200;           // higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller)
    double 	param2 = 10;            // accumulator threshold for the circle centers at the detection stage
    int 	  minRadius = 20;   
    int 	  maxRadius = 100 ;


    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, inv_accumulator,
                 minDist,  // change this value to detect circles with different distances to each other
                 param1, param2, minRadius, maxRadius);
    
    for( size_t i = 0; i < circles.size(); i++ ) {
      
      cv::Vec3i c = circles[i];
      cv::Point center = cv::Point(c[0], c[1]);

      // circle center
      cv::circle( *out_img, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
      // circle outline
      int radius = c[2];
      cv::circle( *out_img, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);      
    }

    return circles.size();
}

void proyection_lines(cv::Mat* out_image, int max_distance, cv::Mat K, cv::Mat RT) {

  float x[2], y[2];
  float side ;
  int b = 0, g = 0, r= 255;
  cv::Scalar color(b,g,r);

  for (int j = 3; j<= max_distance; j++) {
    side = 1;
    for (int i = 0; i<2; i++) {
      float pos_array[4] = {side, 0, (float)j, 1}; // coordenada 3d en el espacio respecto  los ejes de la imagen
      cv::Mat pos(4, 1, CV_32F, pos_array);

      cv::Mat pixel_2d = K * RT * pos;

      float s = pixel_2d.at<float>(2,0);

      x[i] = pixel_2d.at<float>(0,0) /s;
      y[i] = pixel_2d.at<float>(1,0) /s;

      side = -1;
    }
    b += 15;
    g += 40;
    r -= 40;
    color = cv::Scalar(b,g,r);
    cv::line (*out_image, cv::Point(x[0],y[0]), cv::Point(x[1],y[1]), color, 4, cv::LINE_8, 0);
    cv::putText(*out_image, std::to_string(j), cv::Point(x[0]+10,y[0]),cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2, 4,false);
  }
}

void proyect_circle(float x, float y, float z, cv::Mat* out_img, cv::Mat K, cv::Mat RT) {
  
  float z_gap = 0.125, y_gap = 0.9864;

  float pos_array[4] = {x, y + y_gap, z - z_gap, 1};
  cv::Mat pos(4, 1, CV_32F, pos_array);
  cv::Mat pixel_2d = K * RT * pos;
  
  float s = pixel_2d.at<float>(2,0);
  float x2d = pixel_2d.at<float>(0,0) /s;
  float y2d = pixel_2d.at<float>(1,0) /s;
  //std::cout << x2d << " " << y2d << std::endl;
  cv::Point center(x2d, y2d);

  cv::circle( *out_img, center, 3, cv::Scalar(0,0,255), 5, cv::LINE_AA);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::dnn4_v20191202::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame) {
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
        if ( classes[classId].compare("diningtable") == 0){ diningTable = true;}
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}

// Get the names of the output layers
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net) {
    static std::vector<std::string> names;
    if (names.empty()) {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}


/**
  TO-DO
*/
pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  // Create output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;

  // Blue ball detection and removal of outliers
  out_pointcloud = color_filter(in_pointcloud, 85,200, 0.33,1, 0.078,1);

  if (out_pointcloud.size() > 0 ) {
    // convert pointcloudXYZRGB to pointcloudXYZPtr 
    pcl::PointCloud<pcl::PointXYZ>::Ptr aux_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(out_pointcloud, *aux_cloud_ptr);

    // StatisticalOutlierRemoval
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(aux_cloud_ptr);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_filtered);
    pcl::copyPointCloud(*cloud_filtered,out_pointcloud);

    std::vector<std::vector<float> > center_values = extract_model(pcl::SACMODEL_SPHERE, &out_pointcloud, false);

    pthread_mutex_lock(&mutex);
    ball_points = center_values;
    pthread_mutex_unlock(&mutex);

    for ( int i = 0; i< (int)center_values.size() ; i++) {

      float x_ball = center_values[i][0];
      float y_ball = center_values[i][1];
      float z_ball = center_values[i][2];

      pcl::PointCloud<pcl::PointXYZRGB> center_ball = create_cube( 0.1, x_ball,y_ball,z_ball, 0,0,255);
      out_pointcloud += center_ball;

      out_pointcloud += proyection_cubes(8);
    }
  } else {
    if(!diningTable) {
      out_pointcloud = in_pointcloud;
    } else {
      out_pointcloud = in_pointcloud;
    }
/*
    pcl::PointCloud<pcl::PointXYZRGB> walls = color_filter(in_pointcloud, 0,255, 0,0.1, 0,1);
    
    for (int i = 0; i < (int)in_pointcloud.size() ; i++) {
      for (int j = 0; j < (int)walls.size(); j++) {
        if (in_pointcloud.points[i].x != walls.points[i].x && in_pointcloud.points[i].y != walls.points[i].y && in_pointcloud.points[i].z != walls.points[i].z ){
          out_pointcloud.points.push_back(in_pointcloud.points[i]);
        }
      }
    }
    
  
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(out_pointcloud, *cloud_ptr);

    //pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud_ptr));
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_ptr));
    std::vector<int> inliers;

    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
    ransac.setDistanceThreshold (.01);
    ransac.computeModel();
    ransac.getInliers(inliers);

    // copies all inliers of the model computed to another PointCloud
    pcl::copyPointCloud (*cloud_ptr, inliers, out_pointcloud);
    /*
    std::vector<std::vector<float> > center_values = extract_model(pcl::SACMODEL_PLANE, &out_pointcloud, true);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  
    copyPointCloud(out_pointcloud, *cloud);
  
    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);
  
    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);
  
    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  
    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (0.03);
  
    // Compute the features
    ne.compute (*cloud_normals);
  
    // cloud_normals->size () should have the same size as the input cloud->size ()*
    std::cout << cloud_normals->size () << std::endl;
    */
  }
  
  return out_pointcloud;
}
std::vector<std::vector<float> > extract_model(int model, pcl::PointCloud<pcl::PointXYZRGB> *pointcloud, bool show_model) {

    std::vector<std::vector<float> > all_coefficients;

  // detect sphere and its coefficients 
    pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2), cloud_filtered_blob (new pcl::PCLPointCloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_2 (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::toPCLPointCloud2 (*pointcloud, *cloud_blob);

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor2;
    sor2.setInputCloud (cloud_blob);
    sor2.setLeafSize (0.01f, 0.01f, 0.01f);
    sor2.filter (*cloud_filtered_blob);
    // Convert to the templated PointCloud
    pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered_2);

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (model);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.01);

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    int i = 0, nr_points = (int) cloud_filtered_2->size ();

    // While 30% of the original cloud is still there
    while (cloud_filtered_2->size () > 0.3 * nr_points)
    {
      // Segment the largest planar component from the remaining cloud
      seg.setInputCloud (cloud_filtered_2);
      seg.segment (*inliers, *coefficients);

      all_coefficients.push_back(coefficients->values);

      if (inliers->indices.size () == 0)
      {
        std::cerr << "Could not estimate a model for the given dataset." << std::endl;
        break;
      }

      // Extract the inliers
      extract.setInputCloud (cloud_filtered_2);
      extract.setIndices (inliers);
      extract.setNegative (false);
      extract.filter (*cloud_p);

      // Create the filtering object
      extract.setNegative (true);
      extract.filter (*cloud_f);
      cloud_filtered_2.swap (cloud_f);

      if(show_model) {
        pcl::PointCloud<pcl::PointXYZRGB> aux;
        copyPointCloud(*cloud_filtered_2,aux );
        *pointcloud += aux;
      }

      i++;
    }
    
    return all_coefficients;
}


pcl::PointCloud<pcl::PointXYZRGB> proyection_cubes( int N ) {
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  float r = 0,g = 255,b = 0;
  float z_gap = 0.125, y_gap = 0.9864;

  for ( int i = 0 ; i < N-3 ; i++) {
    cloud += create_cube( 0.1, -1,  1-y_gap, i+3-z_gap, r,g,b);
    cloud += create_cube( 0.1, 1, 1-y_gap, i+3-z_gap, r,g,b);
    r += 50;
    g -= 50;
    b += 10;
  }

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> create_cube(float L, float X, float Y, float Z, float r, float g, float b ){

  pcl::PointCloud<pcl::PointXYZRGB> cloud;

  pcl::PointXYZRGB pt;
  for (float x =  -L/2; x < L/2 ; x = x + 0.01) {
    for ( float y = -L/2; y < L/2 ; y = y + 0.01) {
      for (float z = -L/2; z < L/2 ; z = z + 0.01) {
        pt.r = r;
        pt.g = g;
        pt.b = b;

        pt.x = x + X;
        pt.y = y + Y ;
        pt.z = -z + Z ;

        cloud.points.push_back(pt);
      }
    }
  }
  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> color_filter(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud, float h_min, float h_max, float s_min, float s_max, float v_min, float v_max) {
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  
  // filtro de color
  pcl::PointCloud<pcl::PointXYZHSV> hsv_pointcloud_in ;
  pcl::PointCloudXYZRGBtoXYZHSV(in_pointcloud, hsv_pointcloud_in);
  pcl::PointXYZRGB pt_RGB;
  BOOST_FOREACH( const pcl::PointXYZHSV& pt, hsv_pointcloud_in.points)
  {          
    // HSV values for blue ball: min->(85,85,20),max->(100,255,255);
    if ((int)pt.h > h_min && (int)pt.h < h_max && (float)pt.s > s_min && (float)pt.s < s_max && (float)pt.v > v_min && (float)pt.v < v_max ){// s and v [0-1]
      pcl::PointXYZHSVtoXYZRGB(pt,pt_RGB);
      cloud.points.push_back(pt_RGB);
    }
  }

  return cloud;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::executors::SingleThreadedExecutor exec;

  auto cv_node = std::make_shared<ComputerVisionSubscriber>();
  auto pcl_node = std::make_shared<PCLSubscriber>();
  exec.add_node(cv_node);
  exec.add_node(pcl_node);
  exec.spin();
  
  rclcpp::shutdown();
  return 0;
}