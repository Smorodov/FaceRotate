
#include "opencv2/opencv.hpp"
#include <omp.h>
#include <cv.h>
#include "triangle.h"

using namespace std;
using namespace cv;

// --------------------------------------------------------------
// Вычисление габаритного прямоугольника для точек типа Point2d
// --------------------------------------------------------------
cv::Rect_<double> boundingRect(vector<Point2d>& pts)
{
    cv::Rect_<double> r;
    double minx=FLT_MAX,maxx=FLT_MIN,miny=FLT_MAX,maxy=FLT_MIN;

    for(int i=0;i<pts.size();i++)
    {
        double px=pts[i].x;
        double py=pts[i].y;
        if(minx>px){minx=px;}
        if(miny>py){miny=py;}
        if(maxx<px){maxx=px;}
        if(maxy<py){maxy=py;}
    }

    r.x=minx;
    r.y=miny;
    r.width=maxx-minx;
    r.height=maxy-miny;

    return r;
}

using namespace std;
using namespace cv;
// --------------------------------------------------------------
// Создаем разметку точек, по принадлежности к треугольникам
// --------------------------------------------------------------
void DrawLabelsMask(Mat& imgLabel,vector<Point2d>& points,vector<vector<size_t>>& triangles)
{
	for(int i=0;i<triangles.size();i++)
	{
		Point t[3];
		int ind1=triangles[i][0];
		int ind2=triangles[i][1];
		int ind3=triangles[i][2];
		t[0].x=cvRound(points[ind1].x);
		t[0].y=cvRound(points[ind1].y);
		t[1].x=cvRound(points[ind2].x);
		t[1].y=cvRound(points[ind2].y);
		t[2].x=cvRound(points[ind3].x);
		t[2].y=cvRound(points[ind3].y);
		cv::fillConvexPoly(imgLabel, t, 3, cv::Scalar_<int>((i+1)));
	}
}
// --------------------------------------------------------------
// Предварительный расчет коэффициентов преобразования для пар треугольников
// --------------------------------------------------------------
void CalcCoeffs(vector<Point2d>& s_0,vector<Point2d>& s_1, vector<vector<size_t>>& triangles, Mat& Coeffs)
{
	Rect_<double> Bound_0;
	Rect_<double> Bound_1;
	// Вычислили габариты
	Bound_0=boundingRect(s_0);
	Bound_1=boundingRect(s_1);
	// Предварительный расчет коэффициентов преобразования для пар треугольников
	Coeffs=Mat(triangles.size(),6,CV_64FC1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int i=0;i<triangles.size();i++)
	{
		int ind1=triangles[i][0];
		int ind2=triangles[i][1];
		int ind3=triangles[i][2];
		// Исходные точки (откуда берем)
		Point2d t_0[3];
		t_0[0]=s_0[ind1]-Bound_0.tl(); // i
		t_0[1]=s_0[ind2]-Bound_0.tl(); // j
		t_0[2]=s_0[ind3]-Bound_0.tl(); // k
		// Целевые точки (куда кладем)
		Point2d t_1[3];
		t_1[0]=s_1[ind1]-Bound_1.tl(); // i
		t_1[1]=s_1[ind2]-Bound_1.tl(); // j
		t_1[2]=s_1[ind3]-Bound_1.tl(); // k

		double denom=(t_1[0].x * t_1[1].y + t_1[2].y * t_1[1].x - t_1[0].x * t_1[2].y - t_1[2].x * t_1[1].y - t_1[0].y * t_1[1].x + t_1[0].y * t_1[2].x);

		Coeffs.at<double>(i,0)= -(-t_1[2].y * t_0[1].x + t_1[2].y * t_0[0].x + t_1[1].y * t_0[2].x - t_1[1].y * t_0[0].x - t_1[0].y * t_0[2].x + t_1[0].y * t_0[1].x) / denom;
		Coeffs.at<double>(i,1)= -(t_1[2].x * t_0[1].x - t_1[2].x * t_0[0].x - t_1[1].x * t_0[2].x + t_1[1].x * t_0[0].x + t_1[0].x * t_0[2].x - t_1[0].x * t_0[1].x) / denom;
		Coeffs.at<double>(i,2)= -(t_1[2].x * t_1[1].y * t_0[0].x - t_1[2].x * t_1[0].y * t_0[1].x - t_1[1].x * t_1[2].y * t_0[0].x + t_1[1].x * t_1[0].y * t_0[2].x + t_1[0].x * t_1[2].y * t_0[1].x - t_1[0].x * t_1[1].y * t_0[2].x)/denom;
		Coeffs.at<double>(i,3)= -(t_1[1].y * t_0[2].y - t_1[0].y * t_0[2].y - t_1[2].y * t_0[1].y + t_1[2].y * t_0[0].y - t_0[0].y * t_1[1].y + t_0[1].y * t_1[0].y) / denom;
		Coeffs.at<double>(i,4)= -(-t_1[2].x * t_0[0].y + t_1[0].x * t_0[2].y + t_1[2].x * t_0[1].y - t_0[1].y * t_1[0].x - t_1[1].x * t_0[2].y + t_0[0].y * t_1[1].x) / denom;
		Coeffs.at<double>(i,5)= -(t_0[0].y * t_1[1].y * t_1[2].x - t_0[2].y * t_1[0].x * t_1[1].y - t_0[1].y * t_1[0].y * t_1[2].x + t_0[1].y * t_1[0].x * t_1[2].y + t_0[2].y * t_1[0].y * t_1[1].x - t_0[0].y * t_1[1].x * t_1[2].y) / denom;
	}
}

// --------------------------------------------------------------
// Переносит изображение из img с сеткой на точках s_0
// в изображение dst с сеткой на точках s_1
// Сетка задается треугольниками.
// triangles - вектор треугольников.
// Каждый треугольник - 3 индекса вершин.
// --------------------------------------------------------------
void WarpAffine(Mat& img,vector<Point2d>& s_0,vector<Point2d>& s_1, vector<vector<size_t>>& triangles, Mat& dstLabelsMask,Mat& dst)
{
	Rect_<double> Bound_0;
	Rect_<double> Bound_1;

	// ROI (все точки должны лежать в пределах своих изображений)
	// Вычислили габариты
	Bound_0=boundingRect(s_0);
	Bound_1=boundingRect(s_1);	

	Bound_1.width=cvRound(Bound_1.width);
	Bound_1.height=cvRound(Bound_1.height);

	Bound_0.width=cvRound(Bound_0.width);
	Bound_0.height=cvRound(Bound_0.height);

	if(Bound_0.br().x>img.cols-1){Bound_0.width=(double)img.cols-1-Bound_0.x;}
	if(Bound_0.br().y>img.rows-1){Bound_0.height=(double)img.rows-1-Bound_0.y;}


	Mat I_0=img(Bound_0);

	// Переводим координаты точек в систему координат ROI
	for(int i=0;i<s_1.size();i++)
	{
		s_1[i]-=Bound_1.tl();
	}

	// Корректируем границы
	if(Bound_1.x<0)
	{
		Bound_1.x=0;
	}

	if(Bound_1.y<0)
	{
		Bound_1.y=0;
	}

	if(Bound_1.br().x>dst.cols-1)
	{
		Bound_1.width=(double)dst.cols-1-Bound_1.x;
	}

	if(Bound_1.br().y>dst.rows-1)
	{
		Bound_1.height=(double)dst.rows-1-Bound_1.y;
	}


	// Назначаем ROI
	Mat I_1=dst(Bound_1);

	// Предварительный расчет коэффициентов преобразования для пар треугольников
	Mat Coeffs;
	CalcCoeffs(s_0,s_1,triangles,Coeffs);

	// Сканируем изображение и переносим с него точки на шаблон
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int i=0;i<I_1.rows;i++)
	{
		Point2d W(0,0);
		for(int j=0;j<I_1.cols;j++)
		{
			double x=j;
			double y=i;
			int Label=dstLabelsMask.at<int>(i,j)-1;
			if(Label!=(-1))
			{				
				W.x=Coeffs.at<double>(Label,0)*x+Coeffs.at<double>(Label,1)*y+Coeffs.at<double>(Label,2);
				W.y=Coeffs.at<double>(Label,3)*x+Coeffs.at<double>(Label,4)*y+Coeffs.at<double>(Label,5);
				if(cvRound(W.x)>0 && cvRound(W.x)<I_0.cols && cvRound(W.y)>0 && cvRound(W.y)<I_0.rows)
				{
					I_1.at<Vec3b>(i,j)=I_0.at<Vec3b>(cvRound(W.y),cvRound(W.x));
				}
			}
		}
	}
	cv::GaussianBlur(I_1,I_1,Size(3,3),0.5);	
}
