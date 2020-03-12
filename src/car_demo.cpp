#include <argparse.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <src/SiamMask/siammask.h>

#include "ros/ros.h"

#include <niv_comm/DlBbox.h>
#include <sensor_msgs/Image.h>

#define O_WIDTH     644
#define O_HEIGHT    482

unsigned char Img_data[O_WIDTH*O_HEIGHT*3];

struct Bbox{
  float x = 0;
  float y = 0;
  float width = 0;
  float height = 0;
  float img_width = O_WIDTH;
  float img_height = O_HEIGHT;
  float label = 100;
  float by_tracker = 0;
};

Bbox bbox;


void msgCallback_img(const sensor_msgs::Image::ConstPtr& Img){
  for(int i = 0; i<O_HEIGHT*O_WIDTH*3; i++){
    Img_data[i] = Img->data[i];
  }
}

void msgCallback_box(const niv_comm::DlBbox::ConstPtr& box){
  bbox.x = box->x;
  bbox.y = box->y;
  bbox.width = box->width;
  bbox.height = box->height;
  bbox.img_width = box->img_width;
  bbox.img_height = box->img_height;
  bbox.label = box->label;
  bbox.by_tracker = box->by_tracker;
}


void overlayMask(const cv::Mat& src, const cv::Mat& mask, cv::Mat& dst) {
    std::vector<cv::Mat> chans;
    cv::split(src, chans);
    cv::max(chans[2], mask, chans[2]);
    cv::merge(chans, dst);
}

void drawBox(
    cv::Mat& img, const cv::RotatedRect& box, const cv::Scalar& color,
    int thickness = 1, int lineType = cv::LINE_8, int shift = 0
) {
    cv::Point2f corners[4];
    box.points(corners);
    for(int i = 0; i < 4; ++i) {
        cv::line(img, corners[i], corners[(i + 1) % 4], color, thickness, lineType, shift);
    }
}

enum MODE{
  NONE = 0,
  INIT = 1,
  TRACK = 2
} ;

MODE none(cv::Mat& images) {
  if(bbox.label == 3)
    return MODE::INIT;

  cv::imshow("SiamMask", images);
  cv::waitKey(1);
  return MODE::NONE;
}

MODE init(const Bbox& bbox, State& state, SiamMask& siammask, cv::Mat& images, const torch::Device& device){
  if(bbox.label == 0)
  {
    return MODE::NONE;
  }
  std::cout << "Initializing..." << std::endl;
  cv::Rect roi(bbox.x*bbox.img_width, bbox.y*bbox.img_height, bbox.width*bbox.img_width, bbox.height*bbox.img_height);
  siameseInit(state, siammask, images, roi, device);
  cv::rectangle(images, roi, cv::Scalar(0, 255, 0));

  return MODE::TRACK;
}

MODE track(Bbox& bbox, State& state, SiamMask& siammask, cv::Mat& images, const torch::Device& device, niv_comm::DlBbox& pub_box, const ros::Publisher& pub_tracking){
  siameseTrack(state, siammask, images, device);
  overlayMask(images, state.mask, images);
  drawBox(images, state.rotated_rect, cv::Scalar(0, 255, 0));

  pub_box.x = state.rotated_rect.boundingRect2f().x / bbox.img_width;
  pub_box.y = state.rotated_rect.boundingRect2f().y / bbox.img_height;
  pub_box.width = state.rotated_rect.boundingRect2f().width / bbox.img_width;
  pub_box.height = state.rotated_rect.boundingRect2f().height / bbox.img_height;
  pub_box.label = 3;
  pub_box.by_tracker = 1;
  pub_box.img_height = bbox.img_height;
  pub_box.img_width = bbox.img_width;
  pub_tracking.publish(pub_box);

  cv::imshow("SiamMask", images);
  cv::waitKey(1);

  auto area = pub_box.width * pub_box.height;

  if(area < 0.0015)
  {
    bbox.label = 0;
    return MODE::NONE;
  }

  return MODE::TRACK;
}

int main(int argc, const char* argv[]) {
  std::map<std::string, std::string> remaps;

  ros::init(remaps, "car_tracking");
  ros::NodeHandle nh_tracking;
  //ros::Subscriber sub_image = nh_tracking.subscribe("/vision/image", 10, msgCallback_img);
  ros::Subscriber sub_image = nh_tracking.subscribe("/videofile/image_raw", 1, msgCallback_img);
  ros::Subscriber sub_box = nh_tracking.subscribe("/car_detection_dl/info", 1, msgCallback_box);
  ros::Publisher pub_tracking = nh_tracking.advertise<niv_comm::DlBbox>("/car_detection_dl/tracking_box",1);
  ros::Rate loop_rate(20);
  cv::Mat images;


  argparse::ArgumentParser parser;
  parser.addArgument("-m", "--modeldir", 1, false);
  parser.addArgument("-c", "--config", 1, false);

  parser.parse(argc, argv);

  torch::Device device(torch::kCUDA);

  SiamMask siammask(parser.retrieve<std::string>("modeldir"), device);

  State state;
  state.load_config(parser.retrieve<std::string>("config"));

  auto mode = MODE::NONE;

  cv::namedWindow("SiamMask");
  while (ros::ok()){
    images = cv::Mat(O_HEIGHT, O_WIDTH, CV_8UC3, &Img_data);
    cv::resize(images, images, cv::Size(O_WIDTH, O_HEIGHT));
    niv_comm::DlBbox pub_box;

    switch(mode){
      case MODE::NONE:
        mode = none(images);
        break;

      case MODE::INIT:
        mode = init(bbox, state, siammask, images, device);
        break;

      case MODE::TRACK:
        mode = track(bbox, state, siammask, images, device, pub_box, pub_tracking);
        break;
    }

    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}

