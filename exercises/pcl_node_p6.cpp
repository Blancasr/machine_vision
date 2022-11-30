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

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types_conversion.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud);
pcl::PointCloud<pcl::PointXYZRGB> color_filter(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud, float h_min, float h_max, float s_min, float s_max, float v_min, float v_max);
pcl::PointCloud<pcl::PointXYZRGB> create_cube(float L, float x, float y, float z, float r, float g, float b );
pcl::PointCloud<pcl::PointXYZRGB> proyection_cubes( int N );

bool show_all_pc = false;

class ComputerVisionSubscriber : public rclcpp::Node
{
  public:
    ComputerVisionSubscriber()
    : Node("opencv_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      subscription_3d_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/head_front_camera/depth_registered/points", qos, std::bind(&ComputerVisionSubscriber::topic_callback_3d, this, std::placeholders::_1));
    
      publisher_3d_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pcl_points", qos);
    }

  private:

    void topic_callback_3d(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {    
      
      // Convert to PCL data type
      pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
      pcl::fromROSMsg(*msg, point_cloud);     

      /*
      BOOST_FOREACH( const pcl::PointXYZRGB& pt, point_cloud.points)
      {          
          std::cout  << "x: " << pt.x <<"\n";
          std::cout  << "y: " << pt.y <<"\n";
          std::cout  << "z: " << pt.z <<"\n";
          std::cout  << "rgb: " << (int)pt.r << "-" << (int)pt.g << "-" << (int)pt.b <<"\n";
          std::cout << "---------" << "\n";
      }
      */

      pcl::PointCloud<pcl::PointXYZRGB> pcl_pointcloud = pcl_processing(point_cloud);
      
      // Convert to ROS data type
      sensor_msgs::msg::PointCloud2 output;
      pcl::toROSMsg(pcl_pointcloud, output);

      // copia la cabecera de los datos en la salida y evita errores en Rviz2
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
pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  // Create output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;

  // Processing
  out_pointcloud = color_filter(in_pointcloud, 85,200, 0.33,1, 0.078,1);

  // removal of outliers

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

  // si quisiesemos quedarnos con el 'ruido' :
  //sor.setNegative (true);
  //sor.filter (*cloud_filtered);

  pcl::copyPointCloud(*cloud_filtered,out_pointcloud);

/*
  // deteccion de esferas con RandomSampleCosenus

  // en este ejemplo out_pointcloud satisface un modelo de esfera
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(out_pointcloud, *cloud_ptr);

  pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud_ptr));
  //pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(out_pointcloud));
  std::vector<int> inliers;

  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
  ransac.setDistanceThreshold (.01);
  ransac.computeModel();
  ransac.getInliers(inliers);
  
  // copies all inliers of the model computed to another PointCloud
  pcl::copyPointCloud (*cloud_ptr, inliers, out_pointcloud);

*/

// deteccion de esferas y su scoeficientes con SACSegmentation
  pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2), cloud_filtered_blob (new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_2 (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  
  // Fill cloud_blob
  pcl::toPCLPointCloud2 (out_pointcloud, *cloud_blob);

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor2;
  sor2.setInputCloud (cloud_blob);
  sor2.setLeafSize (0.01f, 0.01f, 0.01f);
  sor2.filter (*cloud_filtered_blob);

  // Convert to the templated PointCloud
  pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered_2);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_SPHERE);
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

    if (inliers->indices.size () == 0)
    {
      std::cerr << "Could not estimate a spherical model for the given dataset." << std::endl;
      break;
    }

    //float radius = coefficients->values[3];
    float x_ball = coefficients->values[0];
    float y_ball = coefficients->values[1];
    float z_ball = coefficients->values[2];

    pcl::PointCloud<pcl::PointXYZRGB> center_ball = create_cube( 0.1, x_ball,y_ball,z_ball, 0,0,255);
    //std::cout << x_ball << " " << y_ball << " " << z_ball << std::endl;
    out_pointcloud += center_ball;

    
    // Extract the inliers
    extract.setInputCloud (cloud_filtered_2);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_p);
    //std::cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered_2.swap (cloud_f);
    
    i++;
  }

  // proyecciones de distancias en 3D
  out_pointcloud += proyection_cubes(8);

  if (show_all_pc) {
    out_pointcloud += in_pointcloud;
  }

  return out_pointcloud;

}
pcl::PointCloud<pcl::PointXYZRGB> proyection_cubes( int N ) {
  
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  float r = 0,g = 255,b = 0;
  float z_gap = 0.225, y_gap = 0.9864, x_gap = 0.02;
  //float z_gap = 0.125, y_gap = 0.9864;

  for ( int i = 0 ; i < N-3 ; i++) {
    cloud += create_cube( 0.1, -1 + x_gap,  y_gap, i+3 - z_gap, r,g,b);
    cloud += create_cube( 0.1, 1 + x_gap, y_gap, i+3 - z_gap, r,g,b);
    r += 40;
    g -= 40;
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
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}