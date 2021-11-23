#include <gms_matcher.h>
#include <iostream>
#include <chrono>
/*#include <string>*/
/*#include <nmmintrin.h>*/
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

/*typedef vector<uint32_t> DescType;
   
/*void GmsMatch(Mat &img1, Mat &img2);
void RansacMatch(Mat &img1, Mat &img2);*/
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);
/*void BfMatch(const vector<DescType> &desc1, const vector<DescType> &desc2, vector<cv::DMatch> &matches);
/*void ComputeORB(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors);
void runImagePair(Mat &img1, Mat &img2) {	
void ComputeORB(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors);
       	//time record
       
	 
	GmsMatch(img1, img2);
	RansacMatch(img1, img2);
}*/

  
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

       chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//set time record
       chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
       chrono::duration<double,std::milli> time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);
         vector<KeyPoint> kp1, kp2;
	 Mat d1, d2;
	/* Mat &img1.size();*/
	/*  cv::FAST(img1, kp1, 40);
	  ComputeORB(img1, kp1, d1);
	  cv::FAST(img2, kp2, 40);
	  ComputeORB(img2, kp2, d2);*/
	  
	vector<DMatch> matches_all, matches_gms;
	Ptr<ORB> orb = ORB::create(3000);
	orb->setFastThreshold(0);
        t1 = chrono::steady_clock::now();
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
	t1 = chrono::steady_clock::now();
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
	/*BfMatch(d1, d2, matches_all);*/
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);	
	 cout << "BFmatch cost time: " << time_used.count() << " ms." << endl;
	cout<<"BF get total :"<<matches_all.size()<<"matches"<<endl;
	Mat img_match=DrawInlier(img1, img2, kp1, kp2, matches_all, 1);
	imshow("ORB+cupipei", img_match);
	cout<<matches_all.size()<<endl;

#endif	
	// GMS filter
	t1 = chrono::steady_clock::now();
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
	int num_inliers = gms.GetInlierMask(vbInliers, false, false);
		// collect matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}
	
		
	
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);
	cout << "gms_match cost time: " << time_used.count() << " ms." << endl;
	cout << "Gms get total " << num_inliers << " matches." << endl;
	
	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
        imshow("ORB+Gms", show);
	waitKey();
  
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0) { cuda::setDevice(0); }
#endif // USE_GPU
        
        //RANSAC
        std::vector<DMatch> matches,Ransacmatches;
	matches=matches_gms;//record !!! changed place
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
	for(size_t i1=0;i1<points1.size();i1++)
	{ 
	  if(norm(points2[i1]-points1t.at<Point2f>((int)i1,0))<=ransacReprojTHreshold)
	  {
	  Ransacmatches.push_back(matches[i1]);
	  }
	  
	}
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double,std::milli>>(t2 - t1);
	 cout << "Ransac_match cost time: " << time_used.count() << " ms." << endl;
	Mat match_img2 = DrawInlier(img1, img2, kp1, kp2, Ransacmatches, 1);
	//Mat match_img2=drawMatches(img1,kp1,img2,kp2,matches,match_img2,Scalar(0,0,255),Scalar(255,255,255),1);
        imshow("ORB+Gms+Ransac", match_img2);
	cout<<"ransac get total:"<<Ransacmatches.size()<<endl;
        
        waitKey();
	/*runImagePair(img1,img2);*/
	return 0;
}
// ORB pattern


/*void RansacMatch( Mat &img1, Mat &img2) {
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
	Mat img_match=DrawInlier(img1, img2, kp1, kp2, matches, 1);
	cout<<"gms get total :"<<matches.size()<<"matches"<<endl;
	imshow("ORB+cupipei", img_match);
#endif	
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

	Ptr<ORB> orb = ORB::create(10000);
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
#endif	
	// GMS filter
	std::vector<bool> vbInliers;
	t1 = chrono::steady_clock::now();
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

	// draw matching
	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
	t2 = chrono::steady_clock::now();
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
// brute-force matching
/*void BfMatch(const vector<DescType> &desc1, const vector<DescType> &desc2, vector<cv::DMatch> &matches) {
  const int d_max = 40;

  for (size_t i1 = 0; i1 < desc1.size(); ++i1) {
    if (desc1[i1].empty()) continue;
    cv::DMatch m{i1, 0, 256};
    for (size_t i2 = 0; i2 < desc2.size(); ++i2) {
      if (desc2[i2].empty()) continue;
      int distance = 0;
      for (int k = 0; k < 8; k++) {
        distance += _mm_popcnt_u32(desc1[i1][k] ^ desc2[i2][k]);
      }
      if (distance < d_max && distance < m.distance) {
        m.distance = distance;
        m.trainIdx = i2;
      }
    }
    if (m.distance < d_max) {
      matches.push_back(m);
    }
  }
}
// compute the descriptor
/*void ComputeORB(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors) {
  const int half_patch_size = 8;
  const int half_boundary = 16;
  int bad_points = 0;
  for (auto &kp: keypoints) {
    if (kp.pt.x < half_boundary || kp.pt.y < half_boundary ||
        kp.pt.x >= img.cols - half_boundary || kp.pt.y >= img.rows - half_boundary) {
      // outside
      bad_points++;
      descriptors.push_back({});
      continue;
    }

    float m01 = 0, m10 = 0;
    for (int dx = -half_patch_size; dx < half_patch_size; ++dx) {
      for (int dy = -half_patch_size; dy < half_patch_size; ++dy) {
        uchar pixel = img.at<uchar>(kp.pt.y + dy, kp.pt.x + dx);
        m10 += dx * pixel;
        m01 += dy * pixel;
      }
    }

    // angle should be arc tan(m01/m10);
    float m_sqrt = sqrt(m01 * m01 + m10 * m10) + 1e-18; // avoid divide by zero
    float sin_theta = m01 / m_sqrt;
    float cos_theta = m10 / m_sqrt;

    // compute the angle of this point
    DescType desc(8, 0);
    for (int i = 0; i < 8; i++) {
      uint32_t d = 0;
      for (int k = 0; k < 32; k++) {
        int idx_pq = i * 32 + k;
        cv::Point2f p(ORB_pattern[idx_pq * 4], ORB_pattern[idx_pq * 4 + 1]);
        cv::Point2f q(ORB_pattern[idx_pq * 4 + 2], ORB_pattern[idx_pq * 4 + 3]);

        // rotate with theta
        cv::Point2f pp = cv::Point2f(cos_theta * p.x - sin_theta * p.y, sin_theta * p.x + cos_theta * p.y)
                         + kp.pt;
        cv::Point2f qq = cv::Point2f(cos_theta * q.x - sin_theta * q.y, sin_theta * q.x + cos_theta * q.y)
                         + kp.pt;
        if (img.at<uchar>(pp.y, pp.x) < img.at<uchar>(qq.y, qq.x)) {
          d |= 1 << k;
        }
      }
      desc[i] = d;
    }
    descriptors.push_back(desc);
  }

  cout << "bad/total: " << bad_points << "/" << keypoints.size() << endl;
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


