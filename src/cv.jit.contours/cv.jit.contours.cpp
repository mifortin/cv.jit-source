/*
cv.jit.contours
	

Copyright 2015, Michael Fortin

This file is part of cv.jit.

cv.jit is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published 
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

cv.jit is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with cv.jit.  If not, see <http://www.gnu.org/licenses/>.

*/

/*
This file links to the OpenCV library <http://sourceforge.net/projects/opencvlibrary/>

Please refer to the  Intel License Agreement For Open Source Computer Vision Library.

Please also read the notes concerning technical issues with using the OpenCV library
in Jitter externals.
*/


#ifdef __cplusplus
extern "C" {
#endif
#include "jit.common.h"
#ifdef __cplusplus 
} //extern "C"
#endif

#undef error
#include "cv.h"
#include "jitOpenCV.h"
#include <opencv2/imgproc/imgproc.hpp>

#include <memory>
#include <vector>


typedef struct _cv_jit_contours 
{
	t_object				ob;
	long*					data;			//!< Data in the matrix
	int						numElements;	//!< Elements allocated
} t_cv_jit_contours;



void *_cv_jit_contours_class;

t_jit_err 				cv_jit_contours_init(void); 
t_cv_jit_contours *	cv_jit_contours_new(void);
void 					cv_jit_contours_free(t_cv_jit_contours *x);
t_jit_err 				cv_jit_contours_matrix_calc(t_cv_jit_contours *x, void *inputs, void *outputs);

t_jit_err cv_jit_contours_init(void) 
{
	long attrflags=0;
	t_jit_object *mop,*output,*io;
	
	_cv_jit_contours_class = jit_class_new("cv_jit_contours",(method)cv_jit_contours_new,(method)cv_jit_contours_free,
		sizeof(t_cv_jit_contours),A_CANT,0L); //A_CANT = untyped

	//add mop
	mop = (t_jit_object *)jit_object_new(_jit_sym_jit_mop,1,1);  //Object has one input and one output
	output = (t_jit_object *)jit_object_method(mop,_jit_sym_getoutput,1); //Get a pointer to the output matrix

   	jit_mop_single_type(mop,_jit_sym_float32);   //Set input type and planecount
   	jit_mop_single_planecount(mop,1);
   	
   	jit_mop_output_nolink(mop,1); //Turn off output linking so that output matrix does not adapt to input
	
	//Set input format
	io = (t_jit_object *)jit_object_method(mop,_jit_sym_getinput,1); //Get a pointer to the input matrix
	jit_attr_setlong(io,_jit_sym_minplanecount,1);
  	jit_attr_setlong(io,_jit_sym_maxplanecount,1);
  	jit_attr_setlong(io,_jit_sym_mindim,2);
  	jit_attr_setlong(io,_jit_sym_maxdim,2);
   	
   	jit_attr_setlong(output,_jit_sym_minplanecount,3);  //3 planes:
  	jit_attr_setlong(output,_jit_sym_maxplanecount,3);	//Blob ID, x-coord, y-coord
  	jit_attr_setlong(output,_jit_sym_mindim,1); //Only one dimension
  	jit_attr_setlong(output,_jit_sym_maxdim,1);
  	jit_attr_setsym(output,_jit_sym_types,_jit_sym_long); //Coordinates are returned with sub-pixel accuracy
   	   	
	jit_class_addadornment(_cv_jit_contours_class,mop);
	
	
	//add methods
	jit_class_addmethod(_cv_jit_contours_class, (method)cv_jit_contours_matrix_calc, 		"matrix_calc", 		A_CANT, 0L);	
	
	//add attributes
	attrflags = JIT_ATTR_GET_DEFER_LOW | JIT_ATTR_SET_USURP_LOW;
			
	jit_class_register(_cv_jit_contours_class);

	return JIT_ERR_NONE;
}

t_jit_err cv_jit_contours_matrix_calc(t_cv_jit_contours *x, void *inputs, void *outputs)
{
	t_jit_err err=JIT_ERR_NONE;
	long in1_savelock=0, out_savelock=0;
	t_jit_matrix_info in1_minfo, out_minfo;
	char *in1_bp;
	void *in1_matrix, *out_matrix;
	CvMat in_mat1;
		
	//Get pointers to matrices
	in1_matrix 	= jit_object_method(inputs,_jit_sym_getindex,0);
	out_matrix  = jit_object_method(outputs,_jit_sym_getindex,0);

	if (x&&in1_matrix && out_matrix)
	{
		//Lock the matrices
		in1_savelock = (long) jit_object_method(in1_matrix,_jit_sym_lock,1);
		out_savelock = (long) jit_object_method(out_matrix,_jit_sym_lock,1);
		
		//Make sure input is of proper format
		jit_object_method(in1_matrix,_jit_sym_getinfo,&in1_minfo);
		jit_object_method(out_matrix,_jit_sym_getinfo,&out_minfo);

		if(in1_minfo.dimcount != 2)
		{
			err = JIT_ERR_MISMATCH_DIM;
			goto out;
		}
		if(in1_minfo.planecount != 1)
		{
			err = JIT_ERR_MISMATCH_PLANE;
			goto out;
		}
		
		//Get data
		jit_object_method(in1_matrix,_jit_sym_getdata,&in1_bp);
		if(!in1_bp){
			err = JIT_ERR_INVALID_INPUT;
			goto out;
		}
		
		cvJitter2CvMat(in1_matrix, &in_mat1);
		
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::Mat in_mat1x(&in_mat1);
		cv::findContours(in_mat1x, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		
		//Count the number of elements...
		int total = 0;
		for (auto &pointList : contours)
		{
			total += pointList.size();
		}
		
		if (total*3 > x->numElements)
		{
			delete[] x->data;
			x->data = new long[total*3];
			x->numElements = total*3;
		}
		
		// Fill...
		long *target = x->data;
		int numContour = 0;
		for (auto &pointList : contours)
		{
			for (cv::Point &pt : pointList)
			{
				target[0] = pt.x;
				target[1] = pt.y;
				target[2] = numContour;
				target += 3;
			}
			numContour++;
		}
		
		//Prepare output:
		out_minfo.dim[0] = total;
		out_minfo.dimcount = 1;
		out_minfo.planecount = 3;
		out_minfo.type = _jit_sym_long;
		out_minfo.dimstride[0] = 3 * sizeof(long);
		out_minfo.dimstride[1] = out_minfo.size = out_minfo.dim[0] * out_minfo.dimstride[0];
		out_minfo.flags = JIT_MATRIX_DATA_REFERENCE | JIT_MATRIX_DATA_FLAGS_USE;
		jit_object_method(out_matrix,_jit_sym_setinfo_ex, &out_minfo);
		jit_object_method(out_matrix,_jit_sym_data, x->data);
	}
		
	
out:
	jit_object_method(out_matrix,gensym("lock"),out_savelock);
	jit_object_method(in1_matrix,gensym("lock"),in1_savelock);
	return err;
}



t_cv_jit_contours *cv_jit_contours_new(void)
{
	t_cv_jit_contours *x;
	
	if ((x=(t_cv_jit_contours *)jit_object_alloc(_cv_jit_contours_class))) {
		x->data = new long[1000];
		x->numElements = 1000;
	} else {
		x = NULL;
	}	
	return x;
}

void cv_jit_contours_free(t_cv_jit_contours *x)
{
	if (x->data) delete[] x->data;
}
