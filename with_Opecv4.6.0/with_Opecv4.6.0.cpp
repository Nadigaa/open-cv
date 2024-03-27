#include <iostream>
#include <vector>

#include "opencv2/core.hpp"

#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"



using namespace cv;
using namespace std;


using std::cout;
using std::endl;

int main(int argc, char* argv[])
{
    
    Mat img_object = imread("template.bmp", IMREAD_GRAYSCALE);
    Mat img_scene = imread("scene.bmp", IMREAD_GRAYSCALE);

   

    if (img_object.empty() || img_scene.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
      
        return -1;
    }

    Mat img_object1;
    Mat img_scene1;

    cvtColor(img_object, img_object1, COLOR_GRAY2RGB);
    cvtColor(img_scene, img_scene1, COLOR_GRAY2RGB);

    //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    int minHessian = 100;

    Ptr<SIFT> detector = SIFT::create(minHessian);

    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    

    Mat descriptors_object, descriptors_scene;

    detector->detectAndCompute(img_object, noArray(), keypoints_object, descriptors_object);
    detector->detectAndCompute(img_scene, noArray(), keypoints_scene, descriptors_scene);

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SIFT is a floating-point descriptor NORM_L2 is used

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    std::vector< std::vector<DMatch> > knn_matches;

    matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);
   
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;

    std::vector<DMatch> good_matches;
    std::vector<DMatch> good_matches_inlier;
    std::vector<DMatch> good_matches_outlier;

    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //-- Draw matches
    Mat img_matches;
    Mat inlier_matches;
    Mat outlier_matches;

    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1), 
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

    int nModelPoint_EXT = good_matches.size();

    std::vector<Point2f> obj_inlier;
    std::vector<Point2f> scene_inlier;

    std::vector<Point2f> obj_outlier;
    std::vector<Point2f> scene_outlier;

    vector<unsigned char> Mask;

    Mat H = findHomography(obj, scene, RANSAC, 0.9, Mask);

    
    
    int k = 0, k1 = 0;

    
    for (int i = 0; i < nModelPoint_EXT; i++)
    {
        if (Mask[i] == 1)
        {
            k++;
        }
        else if (Mask[i] < 1)
            k1++;

    }

    obj_inlier.resize(k);
    scene_inlier.resize(k);

    obj_outlier.resize(k1);
    scene_outlier.resize(k1);

    k = 0, k1 = 0;

    // outlier and inlier points after RANSAC

    for (size_t i = 0; i < nModelPoint_EXT; i++)
    {
        if (Mask[i] == 1)
        {
            obj_inlier[k].x = obj[i].x;
            obj_inlier[k].y = obj[i].y;

            scene_inlier[k].x = scene[i].x;
            scene_inlier[k].y = scene[i].y;

                  
            good_matches_inlier.push_back(good_matches[i]);

            k++;
        }
        else if (Mask[i] == 0)
        {
            obj_outlier[k1].x = obj[i].x;
            obj_outlier[k1].y = obj[i].y;

            scene_outlier[k1].x = scene[i].x;
            scene_outlier[k1].y = scene[i].y;

            good_matches_outlier.push_back(good_matches[i]);

            k1++;
        }         

    }



    //draw inlier match
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches_inlier, inlier_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //draw outlier match
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches_outlier, outlier_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    //Draw feature points

    Point pt_obj, pt_scene;
    for (int i = 0; i < obj_inlier.size(); i++)
    {
        pt_obj = Point(cvRound(obj_inlier[i].x), cvRound(obj_inlier[i].y));
        pt_scene = Point(cvRound(scene_inlier[i].x), cvRound(scene_inlier[i].y));

  //      circle(img_object1, pt_obj, 1, Scalar(255, 0, 0), 3, LINE_AA);
  //      circle(img_scene1, pt_scene, 1, Scalar(255, 0, 0), 3, LINE_AA);
        

    }

    //draw outlier

    Point pt_obj1, pt_scene1;
    for (int i = 0; i < obj_outlier.size(); i++)
    {
        pt_obj1 = Point(cvRound(obj_outlier[i].x), cvRound(obj_outlier[i].y));
        pt_scene1 = Point(cvRound(scene_outlier[i].x), cvRound(scene_outlier[i].y));

        circle(img_object1, pt_obj1, 1, Scalar(255, 255, 0), 3, LINE_AA);
        circle(img_scene1, pt_scene1, 1, Scalar(255, 255, 0), 3, LINE_AA);


    }

   
    //-- Show detected matches

    imshow("Good Matches & Object detection", img_matches);

    imshow("Inlier matches", inlier_matches);
    imshow("Outlier matches", outlier_matches);
    
    imshow("Draw inlier and outlier matches in template", img_object1);
    imshow("Draw inlier and outlier matches in scene", img_scene1);


    waitKey();

    return 0;
}
