#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <argparse.hpp>
#include <src/SiamMask/siammask.h>

#include "ros/ros.h"
#include <niv_comm/DlBbox.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;

public:
  cv_bridge::CvImagePtr cv_ptr;

  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/videofile/image_raw", 1, &ImageConverter::imageCb, this);
  }

  ~ImageConverter()
  {
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    try
    {
      ImageConverter::cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    // Draw an example circle on the video stream
    //if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
      //cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
    // Update GUI Window
    //cv::waitKey(3);
  }
};

struct Bbox{
  float x = 0;
  float y = 0;
  float width = 0;
  float height = 0;
  float img_width = 0;
  float img_height = 0;
  float label = 100;
  float by_tracker = 0;
};

Bbox bbox;

bool dirExists(const std::string& path)
{
    struct stat info{};
    if (stat(path.c_str(), &info) != 0)
        return false;
    return info.st_mode & S_IFDIR;
}

void msgCallback_img(const sensor_msgs::Image::ConstPtr& Img){
  //for(int i = 0; i<WIDTH*HEIGHT*3; i++){
  //  Img_data[i] = Img->data[i];
  //}
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


std::vector<std::string> listDir(const std::string& path, const std::vector<std::string>& match_ending)
{
    static const auto ends_with = [](std::string const & value, std::string const & ending) -> bool
    {
        if (ending.size() > value.size()) return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    };

    if(!dirExists(path)) {
        throw std::runtime_error(std::string("Directory not found: ") + path);
    }

    std::vector<std::string> files;
    DIR *dir = opendir(path.c_str());

    if(dir == nullptr)
        return files;

    struct dirent *pdirent;
    while ((pdirent = readdir(dir)) != nullptr) {
        std::string name(pdirent->d_name);
        for(const auto& ending : match_ending){
            if(ends_with(name, ending)) {
                files.push_back(path + "/" + name);
                break;
            }
        }
    }
    closedir(dir);

    return files;
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

int main(int argc, const char* argv[]) {
  bool remap_name = true;
  std::map<std::string, std::string> remaps;
  if (remap_name) remaps["__name"] = "test_node";

  ros::init(remaps, "car_tracking");
  ros::NodeHandle nh_tracking;
  //ros::Subscriber sub_image = nh_tracking.subscribe("/videofile/image_raw", 10, msgCallback_img);
  //ros::Subscriber sub_image = nh_tracking.subscribe("/videofile/image_raw", 10, msgCallback_img);
  ros::Subscriber sub_box = nh_tracking.subscribe("/car_detection_dl/info", 10, msgCallback_box);
  ros::Publisher pub_tracking = nh_tracking.advertise<niv_comm::DlBbox>("/car_detection_dl/tracking_box",100);

  ImageConverter ic;


  argparse::ArgumentParser parser;
  parser.addArgument("-m", "--modeldir", 1, false);
  parser.addArgument("-c", "--config", 1, false);
  //parser.addFinalArgument("target");

  parser.parse(argc, argv);

  torch::Device device(torch::kCUDA);

  //std::string modeldir = "home/nearthlab/catkin_ws/src/siammask/src/models/SiamMask_DAVIS";
  //std::string config = "/home/nearthlab/catkin_ws/src/siammask/src/config_davis.json";

  SiamMask siammask(parser.retrieve<std::string>("modeldir"), device);
  //SiamMask siammask(modeldir, device);
  State state;
  state.load_config(parser.retrieve<std::string>("config"));
  //state.load_config(config);
  //const std::string target_dir = parser.retrieve<std::string>("target");

  //image getting
  //std::vector<std::string> image_files = listDir(target_dir, {"jpg", "png", "bmp"});
  //std::sort(image_files.begin(), image_files.end());

  //std::cout << image_files.size() << " images found in " << target_dir << std::endl;

  while (ros::ok()){
    cv::Mat images = ic.cv_ptr->image;
    /*for(const auto& image_file : image_files) {
        images.push_back(cv::imread(image_file));
    }*/

    cv::namedWindow("SiamMask");

    //Select ROI box
    //cv::Rect roi = cv::selectROI("SiamMask", images.front(), false);


    //if(roi.empty())
    //  return EXIT_SUCCESS;

    //int64 tic = cv::getTickCount();

    cv::Mat& src = images;

    if (bbox.label == 3) {
      std::cout << "Initializing..." << std::endl;
      cv::Rect roi(bbox.x*bbox.img_width, bbox.y*bbox.img_height, bbox.width*bbox.img_width, bbox.height*bbox.img_height);
      siameseInit(state, siammask, src, roi, device);
      cv::rectangle(src, roi, cv::Scalar(0, 255, 0));
    }
    else if (bbox.label == 0) {
      siameseTrack(state, siammask, src, device);
      overlayMask(src, state.mask, src);
      drawBox(src, state.rotated_rect, cv::Scalar(0, 255, 0));
    }
    else{

    }

    cv::imshow("SiamMask", src);
    //toc += cv::getTickCount() - tic;
    cv::waitKey(1);


    //double total_time = toc / cv::getTickFrequency();
    //double fps = image_files.size() / total_time;
    //printf("SiamMask Time: %.1fs Speed: %.1ffps (with visulization!)\n", total_time, fps);
    niv_comm::DlBbox pub_box;
    if (bbox.label != 100)
      pub_box.x = state.rotated_rect.boundingRect2f().x/bbox.img_width;
      pub_box.y = state.rotated_rect.boundingRect2f().y/bbox.img_height;
      pub_box.width = state.rotated_rect.boundingRect2f().width/bbox.img_width;
      pub_box.height = state.rotated_rect.boundingRect2f().height/bbox.img_height;
      pub_box.label = 3;
      pub_box.by_tracker = 1;
      pub_box.img_height = bbox.img_height;
      pub_box.img_width = bbox.img_width;

      pub_tracking.publish(pub_box);

    ros::spin();
    return EXIT_SUCCESS;
  }
}

