#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
#include "triangle.h"
#include "WarpAffine.h"

void WarpAffine(Mat& img,vector<Point2d>& s_0,vector<Point2d>& s_1, vector<vector<size_t> >& triangles, Mat& dstLabelsMask,Mat& dst);

vector <vector<size_t> > Triangulate(vector<Point2f>& pts)
{
	vector<vector<size_t> > triangles;

	struct triangulateio in, out, vorout;

	in.numberofpoints = pts.size();

	in.numberofpointattributes = 0;
	in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
	in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));

	for(int i=0;i<pts.size();i++)
	{
		in.pointlist[2*i] = pts[i].x;
		in.pointlist[2*i+1] = pts[i].y;
		in.pointmarkerlist[i]=0;
	}
	in.numberofsegments = 0;
	in.numberofholes = 0;
	in.numberofregions = 0;

	out.pointlist = (REAL *) NULL;
	out.pointattributelist = (REAL *) NULL;
	out.pointmarkerlist = (int *) NULL;
	out.trianglelist = (int *) NULL;
	out.triangleattributelist = (REAL *) NULL;
	out.neighborlist = (int *) NULL;
	out.segmentlist = (int *) NULL;
	out.segmentmarkerlist = (int *) NULL;
	out.edgelist = (int *) NULL;
	out.edgemarkerlist = (int *) NULL;
	vorout.pointlist = (REAL *) NULL;
	vorout.pointattributelist = (REAL *) NULL;
	vorout.edgelist = (int *) NULL;
	vorout.normlist = (REAL *) NULL;


	triangulate("pczeQ", &in, &out, &vorout);

	for (int i = 0; i < out.numberoftriangles; i++) 
	{
		vector<size_t> idx(3);
		for (int j = 0; j < out.numberofcorners; j++) 
		{
			idx[j]=out.trianglelist[i * out.numberofcorners + j];
		}
		triangles.push_back(idx);
	}

	free(in.pointlist);
	free(in.pointmarkerlist);

	free(out.pointlist);
	free(out.pointattributelist);
	free(out.trianglelist);
	free(out.triangleattributelist);

	return triangles;
}

//---------------------------------------------
//
//---------------------------------------------	
int main( int, char** )
{	
	//Points of Female
	int array[] = {76,223,79,262,84,295,91,331,105,359,125,378,150,395,181,403,213,397,237,382,259,365,275,338,282,300,288,265,289,226,266,200,247,191,227,192,209,204,229,200,246,198,98,201,118,191,140,193,161,205,139,201,118,198,116,226,135,215,155,228,135,235,137,223,250,226,232,216,212,228,232,234,233,225,172,227,168,267,156,287,165,299,183,305,204,299,211,292,197,265,194,226,169,294,198,295,144,333,159,327,173,325,183,325,192,325,206,328,222,334,209,343,198,347,183,349,168,347,155,342,165,336,182,338,199,336,199,333,182,333,165,333,183,335,184,286,125,220,145,222,145,231,125,230,241,221,222,222,222,231,241,230};
	std::vector<int> v;
	vector<vector< cv:: Point2d> > contours;
	vector <cv:: Point2d> xy;

	// End Of Declaration for Mask.png 

	string win = "Delaunay Demo";
	namedWindow(win);
	Rect rect(0, 0, 600, 600);
	Mat img(600,600, CV_8UC3);
	img = imread ( ".\\1.jpg" );
	vector<Point2d> s_0;
	vector<Point2d> s_1;

	vector<Point2f> pts;

	//Push into vector "V"

	for (int i = 0; i <= 151; i++) {
		//                cout << array [i] << ",";
		v.push_back(array [i]);
	}

// Add some boundary points
	double dx=(double)img.cols/19.0;
	double dy=(double)img.rows/19.0;
	double x=0,y=0;
	for (int i = 0; i < 20; i++) {
		v.push_back(x);
		v.push_back(0);

		v.push_back(x);
		v.push_back(img.rows-1);

		v.push_back(0);
		v.push_back(y);

		v.push_back(img.cols-1);
		v.push_back(y);
		x+=dx;
		y+=dy;
	}



	for( int i = 0; i < v.size(); i+=2 )
	{
		Point2f fp ( v[i], v [i+1] );
		pts.push_back(fp);
	}

	int count = 82;
	Mat mask(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));

	vector < vector<size_t> > triangles;

	triangles=Triangulate(pts);


s_0.clear();
for ( int i = 0; i < pts.size(); i++ ) {
	s_0.push_back ( pts[i]);
}

	int k=0;
	float t=0;
	namedWindow("destination");
	while(k!=27)
	{
	t+=0.2;
	
	s_1.clear();
	for ( int i = 0; i < pts.size(); i++ ) {

		//s_0.push_back ( pts[i]);
		if(i<76)
		{
			s_1.push_back ( Point ( pts[i].x - 10.0*sin(t), pts[i].y+10.0*cos(0.3*t) ) ); // Transformed points
		}
		else
		{
			s_1.push_back ( Point ( pts[i].x, pts[i].y ) );
		}		
	}
	Rect_<double> Bound=boundingRect(s_1);
	// Смещение в сторону нам не нужно
	for(int i=0;i<s_1.size();i++)
	{
		s_1[i]-=Bound.tl();
	}

	Mat dstLabelsMask=Mat::zeros(Bound.size(),CV_32SC1);
	
	Mat dst(dstLabelsMask.size(), CV_8UC3);

	DrawLabelsMask(dstLabelsMask,s_1,triangles);
	WarpAffine( img, s_0, s_1, triangles,  dstLabelsMask, dst );
	imshow ( "destination", dst);
	k=waitKey(5);
	}
	destroyAllWindows();
	return 0;
}