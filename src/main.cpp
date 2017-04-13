#include<opencv2\opencv.hpp>
#include<iostream>
#include<vector>
#include<algorithm>

#include"SkinDetector.h"
//#include "CurveCSS.h"

#define AREA_THRES 2000
#define CUR_THRES 0.2

using namespace std;
using namespace cv;

double dist(Point x, Point y);

pair<Point, double> circleFromPoints(Point p1, Point p2, Point p3);

/* 1st and 2nd derivative of 1D gaussian
*/
void getGaussianDerivs(double sigma, int M, vector<double>& gaussian, vector<double>& dg, vector<double>& d2g) {
	int L = (M - 1) / 2;
	double sigma_sq = sigma * sigma;
	double sigma_quad = sigma_sq*sigma_sq;
	dg.resize(M); d2g.resize(M); gaussian.resize(M);

	Mat_<double> g = getGaussianKernel(M, sigma, CV_64F);
	for (double i = -L; i < L + 1.0; i += 1.0) {
		int idx = (int)(i + L);
		gaussian[idx] = g(idx);
		// from http://www.cedar.buffalo.edu/~srihari/CSE555/Normal2.pdf
		dg[idx] = (-i / sigma_sq) * g(idx);
		d2g[idx] = (-sigma_sq + i*i) / sigma_quad * g(idx);
	}
}

/* 1st and 2nd derivative of smoothed curve point */
void getdX(vector<double> x,
	int n,
	double sigma,
	double& gx,
	double& dgx,
	double& d2gx,
	vector<double> g,
	vector<double> dg,
	vector<double> d2g,
	bool isOpen = false)
{
	int L = (g.size() - 1) / 2;

	gx = dgx = d2gx = 0.0;
	for (int k = -L; k < L + 1; k++) {
		double x_n_k;
		if (n - k < 0) {
			if (isOpen) {
				x_n_k = x[-(n - k)];
			}
			else {
				x_n_k = x[x.size() + (n - k)];
			}
		}
		else if (n - k > x.size() - 1) {
			if (isOpen) {
				x_n_k = x[n + k];
			}
			else {
				x_n_k = x[(n - k) - (x.size())];
			}
		}
		else {
			x_n_k = x[n - k];
		}
		gx += x_n_k * g[k + L]; //gaussians go [0 -> M-1]
		dgx += x_n_k * dg[k + L];
		d2gx += x_n_k * d2g[k + L];
	}
}

/* 0th, 1st and 2nd derivatives of whole smoothed curve */
void getdXcurve(vector<double> x,
	double sigma,
	vector<double>& gx,
	vector<double>& dx,
	vector<double>& d2x,
	vector<double> g,
	vector<double> dg,
	vector<double> d2g,
	bool isOpen = false)
{
	gx.resize(x.size());
	dx.resize(x.size());
	d2x.resize(x.size());
	for (int i = 0; i<x.size(); i++) {
		double gausx, dgx, d2gx;
		getdX(x, i, sigma, gausx, dgx, d2gx, g, dg, d2g, isOpen);
		gx[i] = gausx;
		dx[i] = dgx;
		d2x[i] = d2gx;
	}
}

void PolyLineSplit(const vector<Point>& pl, vector<double>& contourx, vector<double>& contoury) {
	contourx.resize(pl.size());
	contoury.resize(pl.size());

	for (int j = 0; j<pl.size(); j++)
	{
		contourx[j] = (pl[j].x);
		contoury[j] = (pl[j].y);
	}
}

void PolyLineMerge(vector<Point>& pl, const vector<double>& contourx, const vector<double>& contoury) {
	assert(contourx.size() == contoury.size());
	pl.resize(contourx.size());
	for (int j = 0; j<contourx.size(); j++) {
		pl[j].x = (int) (contourx[j]);
		pl[j].y = (int) (contoury[j]);
	}
}

void ComputeCurveCSS(const vector<double>& curvex,
	const vector<double>& curvey,
	vector<double>& kappa,
	vector<double>& smoothX, vector<double>& smoothY,
	double sigma,
	bool isOpen
)
{
	int M = round((10.0*sigma + 1.0) / 2.0) * 2 - 1;
	assert(M % 2 == 1); //M is an odd number

	vector<double> g, dg, d2g; getGaussianDerivs(sigma, M, g, dg, d2g);

	vector<double> X, XX, Y, YY;
	getdXcurve(curvex, sigma, smoothX, X, XX, g, dg, d2g, isOpen);
	getdXcurve(curvey, sigma, smoothY, Y, YY, g, dg, d2g, isOpen);

	kappa.resize(curvex.size());
	for (int i = 0; i<curvex.size(); i++) {
		// Mokhtarian 02' eqn (4)
		kappa[i] = (X[i] * YY[i] - XX[i] * Y[i]) / pow(X[i] * X[i] + Y[i] * Y[i], 1.5);
	}
}

void ComputeCurveCSS(const vector<Point>& curve,
	vector<double>& kappa,
	vector<Point>& smooth,
	double sigma,
	bool isOpen = false
)
{
	vector<double> contourx(curve.size()), contoury(curve.size());
	PolyLineSplit(curve, contourx, contoury);

	vector<double> smoothx, smoothy;
	ComputeCurveCSS(contourx, contoury, kappa, smoothx, smoothy, sigma, isOpen);

	PolyLineMerge(smooth, smoothx, smoothy);
}



bool findconvex(const vector<double> &kappa, int startidx, int endidx, int &idx)
{
	bool found = false;
	double minval = 0;
	int half_win = 1;

	if (startidx<0 || endidx>kappa.size() - 1 || endidx - startidx < 2 * half_win)
		return false;

	double sum = 0;
	for (int i = startidx; i <= startidx + half_win * 2; i++)
		sum += kappa[i];

	for (int i = startidx+half_win; i<=endidx-half_win; i++) 
	{
		if ( kappa[i] < minval
			 && sum / (2 * half_win + 1) < -(CUR_THRES) // average curvature
			)
		{
			minval = kappa[i];
			idx = i;
			found = true;
		}

		if (i < endidx - half_win)
			sum += kappa[i + half_win + 1] - kappa[i - half_win];
	}
	return found;
}

bool findconcave(const vector<double> &kappa, int startidx, int endidx, int &idx)
{
	bool found = false;
	double maxval = 0;
	int half_win = 1;

	if (startidx<0 || endidx>kappa.size() - 1 || endidx - startidx < 2 * half_win)
		return false;

	double sum = 0;
	for (int i = startidx; i <= startidx + half_win * 2; i++)
		sum += kappa[i];

	for (int i = startidx + half_win; i <= endidx - half_win; i++)
	{
		if (kappa[i] > maxval
			&& sum / (2 * half_win + 1) > (CUR_THRES) // average curvature
			)
		{
			maxval = kappa[i];
			idx = i;
			found = true;
		}

		if (i < endidx - half_win)
			sum += kappa[i + half_win + 1] - kappa[i - half_win];
	}
	return found;
}

int main()
{
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)

	capture.open(0);

	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	Mat cameraFeed;

	SkinDetector mySkinDetector;

	Mat skinMat;

	vector<pair<Point, double> > palm_centers;

	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop
	while (1) {
		//store image to matrix
		capture.read(cameraFeed);
		//cameraFeed = imread("D:/image.png");

		//show the current image
		Mat smallImage;
		//imshow("Original Image", cameraFeed);
		pyrDown(cameraFeed, smallImage);

		skinMat = mySkinDetector.getSkin(smallImage);

		vector<vector<Point> > contours;

		//Enhance edges in the foreground by applying erosion and dilation
		erode(skinMat, skinMat, Mat());
		dilate(skinMat, skinMat, Mat());

		imshow("Skin Image", skinMat);
		//Find the contours in the foreground
		findContours(skinMat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for (int i = 0; i<contours.size(); i++)
			//Ignore all small insignificant areas
			if (contourArea(contours[i]) >= AREA_THRES)
			{
				//Draw contour
				vector<vector<Point> > tcontours;
				tcontours.push_back(contours[i]);
				//drawContours(smallImage, tcontours, -1, cv::Scalar(0, 0, 255), 2);
				
				//Detect Hull in current contour
				vector<vector<Point> > hulls(1); // convex hull points
				vector<vector<int> > hullsI(1); // indices of convex hull points
				convexHull(Mat(tcontours[0]), hulls[0], false);
				convexHull(Mat(tcontours[0]), hullsI[0], false);
				//drawContours(smallImage, hulls, -1, cv::Scalar(0, 255, 0), 2);


				//Find minimum area rectangle to enclose hand
				RotatedRect rect = minAreaRect(Mat(tcontours[0]));

				//Find Convex Defects
				vector<Vec4i> defects;
				if (hullsI[0].size()>0)
				{
					Point2f rect_points[4]; rect.points(rect_points);
					for (int j = 0; j < 4; j++)
						line(smallImage, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 0, 0), 1, 8);
					Point rough_palm_center;
					convexityDefects(tcontours[0], hullsI[0], defects);

					if (defects.size() >= 3)
					{
						// 0- old method based on palm center 
						// 1- new method based on curvature;
						bool finger_method = 1;
						
						if (finger_method==0)
						{

							vector<Point> palm_points;
							for (int j = 0; j < defects.size(); j++)
							{
								int startidx = defects[j][0]; Point ptStart(tcontours[0][startidx]);
								int endidx = defects[j][1]; Point ptEnd(tcontours[0][endidx]);
								int faridx = defects[j][2]; Point ptFar(tcontours[0][faridx]);
								//Sum up all the hull and defect points to compute average
								rough_palm_center += ptFar + ptStart + ptEnd;
								palm_points.push_back(ptFar);
								palm_points.push_back(ptStart);
								palm_points.push_back(ptEnd);
							}

							//Get palm center by 1st getting the average of all defect points, this is the rough palm center,
							//Then U chose the closest 3 points ang get the circle radius and center formed from them which is the palm center.
							rough_palm_center.x /= defects.size() * 3;
							rough_palm_center.y /= defects.size() * 3;
							Point closest_pt = palm_points[0];
							vector<pair<double, int> > distvec;
							for (int i = 0; i < palm_points.size(); i++)
								distvec.push_back(make_pair(dist(rough_palm_center, palm_points[i]), i));
							sort(distvec.begin(), distvec.end());

							//Keep choosing 3 points till you find a circle with a valid radius
							//As there is a high chance that the closes points might be in a linear line or too close that it forms a very large circle
							pair<Point, double> soln_circle;
							for (int i = 0; i + 2 < distvec.size(); i++)
							{
								Point p1 = palm_points[distvec[i + 0].second];
								Point p2 = palm_points[distvec[i + 1].second];
								Point p3 = palm_points[distvec[i + 2].second];
								soln_circle = circleFromPoints(p1, p2, p3);//Final palm center,radius
								if (soln_circle.second != 0)
									break;
							}

							//Find avg palm centers for the last few frames to stabilize its centers, also find the avg radius
							palm_centers.push_back(soln_circle);
							if (palm_centers.size() > 10)
								palm_centers.erase(palm_centers.begin());

							Point palm_center;
							double radius = 0;
							for (int i = 0; i < palm_centers.size(); i++)
							{
								palm_center += palm_centers[i].first;
								radius += palm_centers[i].second;
							}
							palm_center.x /= palm_centers.size();
							palm_center.y /= palm_centers.size();
							radius /= palm_centers.size();

							//Draw the palm center and the palm circle
							//The size of the palm gives the depth of the hand
							circle(smallImage, palm_center, 5, Scalar(144, 144, 255), 3);
							circle(smallImage, palm_center, radius, Scalar(144, 144, 255), 2);


							//Detect fingers by finding points that form an almost isosceles triangle with certain thesholds
							int no_of_fingers = 1;
							for (int j = 0; j < defects.size(); j++)
							{
								int startidx = defects[j][0]; Point ptStart(tcontours[0][startidx]);
								int endidx = defects[j][1]; Point ptEnd(tcontours[0][endidx]);
								int faridx = defects[j][2]; Point ptFar(tcontours[0][faridx]);
								//X o--------------------------o Y
								double Xdist = sqrt(dist(palm_center, ptFar));
								double Ydist = sqrt(dist(palm_center, ptStart));
								double length = sqrt(dist(ptFar, ptStart));
								//line(smallImage, ptEnd, ptFar, Scalar(255, 0, 0), 2);
								//line(smallImage, ptStart, ptFar, Scalar(255, 0, 0), 2);

								double retLength = sqrt(dist(ptEnd, ptFar));
								//Play with these thresholds to improve performance
								if (length <= 3 * radius && Ydist >= 0.4*radius && length >= 40 && retLength >= 10 && max(length, retLength) / min(length, retLength) >= 0.8)
									if (min(Xdist, Ydist) / max(Xdist, Ydist) <= 0.8)
									{
										if ((Xdist >= 0.1*radius && Xdist <= 1.3*radius && Xdist < Ydist) || (Ydist >= 0.1*radius && Ydist <= 1.3*radius && Xdist > Ydist))
										{
											//			line(smallImage, ptEnd, ptFar, Scalar(0, 0, 255), 2);
											//			line(smallImage, ptStart, ptFar, Scalar(0, 0, 255), 2);
											no_of_fingers++;
											char temp[64];
											sprintf(temp, "%d", no_of_fingers);
											putText(smallImage, temp, Point(ptFar.x - 15, ptFar.y + 15), cv::FONT_HERSHEY_PLAIN, 1.5f, cv::Scalar(0, 255, 0), 2);
										}
									}
							}

							no_of_fingers = min(5, no_of_fingers);

							if (no_of_fingers == 0) {
								cout << "click!!" << endl;
							}
							else {
								cout << no_of_fingers << " fingers" << endl;
							}
						} //finger_method==0
						else if (finger_method == 1)
						{
							// re-order contour (from the bottom convex hull point)
							int bottom_index = 0;
							int max_y = 0;
							for (int j = 0; j < hullsI[0].size(); j++)
							{
								int y = tcontours[0][hullsI[0][j]].y;
								if (y > max_y)
								{
									max_y = y;
									bottom_index = hullsI[0][j];
								}
							}
							// re
							
							vector<Point> rcontour;
							rcontour.reserve(tcontours[0].size());
							rcontour.insert(rcontour.end(), tcontours[0].begin() + bottom_index, tcontours[0].end());
							rcontour.insert(rcontour.end(), tcontours[0].begin(), tcontours[0].begin() + bottom_index);


							// Compute curvature:
							//vector<double> contourx, contoury, kappa, smoothx, smoothy;
							vector<Point> smooth;
							vector<double> kappa;
							double sigma = 5.0;
							ComputeCurveCSS(rcontour, kappa,smooth, sigma);
/*							FILE *fp = fopen("D:/data/test.txt", "w");
							for (int j = 0; j < smooth.size(); j++)
							fprintf(fp, "%d %d %d %d %f\n", rcontour[j].x, rcontour[j].y, smooth[j].x, smooth[j].y, kappa[j]);
							fclose(fp);
							imwrite("D:/image.png", cameraFeed);
*/
							// visualize curvature for debugging
							for (int j = 0; j < smooth.size()-1; j++)
							{
								if (kappa[j] < 0)
								{
									line(smallImage, smooth[j], smooth[j + 1], Scalar(0, 0, -kappa[j] * 255), 4);
								}
								else {
									line(smallImage, smooth[j], smooth[j + 1], Scalar(kappa[j] * 255, 0, 0), 4);
								}
							}

							vector<Point> convex_points;
							vector<Point> concave_points;
							int num_contour_pts = smooth.size();
							int num_defects = defects.size();
							// Find all the defect points
							vector<int> defect_idx;
							for (int j = 0; j < defects.size(); j++)
							{
								if (defects[j][2] >= bottom_index)
									defect_idx.push_back(defects[j][2]- bottom_index);
								else
									defect_idx.push_back((num_contour_pts-bottom_index)-1+defects[j][2]);
							}
							sort(defect_idx.begin(), defect_idx.end());
							// Find concave points in segments
							for (int j = 0; j < defect_idx.size(); j++)
							{
								int concave_idx;
								concave_idx = defect_idx[j];
								if (findconcave(kappa, max(0, concave_idx - 4),
									min(num_contour_pts-1, concave_idx + 4), concave_idx))
								concave_points.push_back(smooth[concave_idx]);
							}

							// Find convex points 
							int convex_idx;
							for (int j = 4; j < kappa.size()-4; )
							{
								if (kappa[j]<-(CUR_THRES) && kappa[j] > kappa[j - 1] && kappa[j] < kappa[j + 1])
								{
									if ((findconvex(kappa, j - 4, j + 4, convex_idx)))
									{
										convex_points.push_back(smooth[convex_idx]);
										j += 8; // skip 8 points
									}
								}
								j++;
							}

							// remove point close to image boundary
							for (int j = 0; j < convex_points.size(); )
							{
								if (smallImage.rows - convex_points[j].y < 10
									//smallImage.cols - convex_points[j].x < 10
									)
								{
									convex_points.erase(convex_points.begin() + j);
								}
								else {
									j++;
								}
							}
							for (int j = 0; j < convex_points.size(); j++){
								circle(smallImage, convex_points[j], 5, Scalar(0, 0, 255), 3);
							}
							for (int j = 0; j < concave_points.size(); j++)
								circle(smallImage, concave_points[j], 5, Scalar(255, 0, 0), 3);
						}

					} //if (defects.size() >= 3)
					
				}// (hullsI[0].size()>0) 

			}

		imshow("Frame", smallImage);

		waitKey(30);
	}
	return 0;
}

double dist(Point x, Point y)
{
	return (x.x - y.x)*(x.x - y.x) + (x.y - y.y)*(x.y - y.y);
}

pair<Point, double> circleFromPoints(Point p1, Point p2, Point p3)
{
	double offset = pow(p2.x, 2) + pow(p2.y, 2);
	double bc = (pow(p1.x, 2) + pow(p1.y, 2) - offset) / 2.0;
	double cd = (offset - pow(p3.x, 2) - pow(p3.y, 2)) / 2.0;
	double det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x)* (p1.y - p2.y);
	double TOL = 0.0000001;
	if (abs(det) < TOL) {
		//cout << "POINTS TOO CLOSE" << endl; 
		return make_pair(Point(0, 0), 0);
	}

	double idet = 1 / det;
	double centerx = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet;
	double centery = (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet;
	double radius = sqrt(pow(p2.x - centerx, 2) + pow(p2.y - centery, 2));

	return make_pair(Point(centerx, centery), radius);
}