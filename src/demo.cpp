#include "gms_matcher.h"
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
//#define USE_GPU 
#ifdef USE_GPU
#include <opencv2/cudafeatures2d.hpp>
using cuda::GpuMat;
#endif
using namespace std;
using namespace cv;

   
void GmsMatch(Mat &img1, Mat &img2);
void RansacMatch(Mat &img1, Mat &img2);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);

void runImagePair(Mat &img1, Mat &img2) {	
        /*Mat img1 = imread("../data/2.1.png");//input the image
	Mat img2 = imread("../data/2.2.png");*/
       	//time record
       
	 
	GmsMatch(img1, img2);
	RansacMatch(img1, img2);
}

  
int main( int argc, char** argv )
{
    if ( argc != 3 )
    {
        cout<<"usage:./build/gms_match_demo path_to_img1 path_to_img2"<<endl;
        return 1;
    }
   
    Mat img1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    assert(img1.data && img2.data && "Can not load images!");
  

  
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0) { cuda::setDevice(0); }
#endif // USE_GPU

	runImagePair(img1,img2);
	return 0;
}
void RansacMatch( Mat &img1, Mat &img2) {
	//time record
       chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
       chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
       chrono::duration<double,std::milli> time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);
        t1 = chrono::steady_clock::now();      
        std::vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	std::vector<DMatch> matches,Ransacmatches;

	Ptr<ORB> orb = ORB::create(200);
	orb->setFastThreshold(0);

	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);	
	
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);
	 cout << "select orb cost time: " << time_used.count() << " ms." << endl;
#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches);
	Mat img_match=DrawInlier(img1, img2, kp1, kp2, matches, 1);;
	cout<<"gms get total :"<<matches.size()<<"matches"<<endl;
	imshow("ORB+cupipei", img_match);
#endif	
	// RANSAC filter
	
	/*BFMatcher matcher(NORM_HAMMING,true);
	Vector<DMatch>matches;
	matches.match(d1,d2,matches);*/
        
	    //save the query
	/*std::vector<int>queryIdxs(matches.size()),trainIdxs(matches.size());
	for(size_t i =0;i<matches.size();i++)
	  
	{
	  queryIdxs[i]=matches[i].queryIdx;
	  trainIdxs[i]=matches[i].trainIdx;
	
	}*/
	t1 = chrono::steady_clock::now();
	Mat H12;

	vector<Point2f>points1;
	vector<Point2f>points2;
         for (int i = 0; i < (int) matches.size(); i++) {
         points1.push_back(kp1[matches[i].queryIdx].pt);
         points2.push_back(kp2[matches[i].trainIdx].pt);
  }
	
	int ransacReprojTHreshold=5;
	
	
	H12=findHomography(Mat(points1),Mat(points2),CV_RANSAC,ransacReprojTHreshold);
	vector<char>matchesMask(matches.size(),0);
	Mat points1t;
	perspectiveTransform(Mat(points1),points1t,H12);
	int a=0;
	for(size_t i1=0;i1<points1.size();i1++)
	{ 
	  if(norm(points2[i1]-points1t.at<Point2f>((int)i1,0))<=ransacReprojTHreshold)
	  {
	  Ransacmatches.push_back(matches[i1]);
	  a++;
	  }
	  
	}
	Mat match_img2 = DrawInlier(img1, img2, kp1, kp2, Ransacmatches, 1);
	/*drawMatches(img1,kp1,img2,kp2,matches,match_img2,Scalar(0,0,255),Scalar(255,255,255),matchesMask);*/

	cout<<"ransac get total:"<<a<<"matches"<<endl;
	//draw the position
	/*
	vector<Point2f>img1_corners(4);
	img1_corners[0]=cvPoint(0,0);
	img1_corners[1]=cvPoint(img1.cols,0);
	img1_corners[2]=cvPoint(img1.cols,img1.rows);
	img1_corners[3]=cvPoint(0,img1.rows);
	
	vector<Point2f>img2_corners(4);
	perspectiveTransform(img1_corners,img2_corners,H12);
	line(match_img2,img2_corners[0]+Point2f(static_cast<float>(img1.cols),0),img2_corners[1]+Point2f(static_cast<float>(img1.cols),0),Scalar(0,0,255),2);
	line(match_img2,img2_corners[1]+Point2f(static_cast<float>(img1.cols),0),img2_corners[2]+Point2f(static_cast<float>(img1.cols),0),Scalar(0,0,255),2);
	line(match_img2,img2_corners[2]+Point2f(static_cast<float>(img1.cols),0),img2_corners[3]+Point2f(static_cast<float>(img1.cols),0),Scalar(0,0,255),2);
	line(match_img2,img2_corners[3]+Point2f(static_cast<float>(img1.cols),0),img2_corners[0]+Point2f(static_cast<float>(img1.cols),0),Scalar(0,0,255),2);
        */
	// draw matching
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);
	 cout << "Ransac_match cost time: " << time_used.count() << " ms." << endl;
	imshow("ORB+Ransac", match_img2);
	waitKey();
}




void GmsMatch(Mat &img1, Mat &img2) {
	//time record
       chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
       chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
       chrono::duration<double,std::milli> time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);
       t1 = chrono::steady_clock::now();
        vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;

	Ptr<ORB> orb = ORB::create(20000);
	orb->setFastThreshold(0);

	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);	
	
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);
	 cout << "select orb cost time: " << time_used.count() << " ms." << endl;
       
#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
	cout<<"gms get total :"<<matches_all.size()<<"matches"<<endl;
#endif	
	// GMS filter
	t1 = chrono::steady_clock::now();
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
	int num_inliers = gms.GetInlierMask(vbInliers, false, false);
	cout << "Get total " << num_inliers << " matches." << endl;

	// collect matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}
        t2 = chrono::steady_clock::now();
	// draw matching	
	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
	time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);
	 cout << "gms_match cost time: " << time_used.count() << " ms." << endl;
	imshow("ORB+Gms", show);
	waitKey();
}
  //RANSAC
  /*void Ransac(){
	const int N = mvMatches12.size();

       // Indices for minimum set selection
       vector<size_t> vAllIndices;
       vAllIndices.reserve(N);
       vector<size_t> vAvailableIndices;

       for(int i=0; i<N; i++)     //compute size_t
        {
        vAllIndices.push_back(i);
        }
	 mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

         DUtils::Random::SeedRandOnce(0);

         for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();  //delete 
        }
    }

  }
*/
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
	const int height = max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
	src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(0, 255, 255));
		}
	}
	else if (type == 2)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(255, 0, 0));
		}

		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			circle(output, left, 1, Scalar(0, 255, 255), 2);
			circle(output, right, 1, Scalar(0, 255, 0), 2);
		}
	}

	return output;
}
