/*
max.cv.jit.label
	

Copyright 2010, Jean-Marc Pelletier
jmp@jmpelletier.com

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

#ifdef __cplusplus
extern "C" {
#endif
#include "jit.common.h"
#include "max.jit.mop.h"
#ifdef __cplusplus 
} //extern "C"
#endif

typedef struct _max_cv_jit_label 
{
	t_object		ob;
	void			*obex;
} t_max_cv_jit_label;

t_jit_err cv_jit_label_init(void); 

void *max_cv_jit_label_new(t_symbol *s, long argc, t_atom *argv);
void max_cv_jit_label_free(t_max_cv_jit_label *x);
void *max_cv_jit_label_class;

#ifdef __cplusplus
extern "C"
#endif		 	
int main(void)
{	
	void *p,*q;
	
	union { void **v_ptr; t_messlist **m_ptr; } alias_ptr;
	alias_ptr.v_ptr = &max_cv_jit_label_class;
	cv_jit_label_init();
	setup(alias_ptr.m_ptr, (method)max_cv_jit_label_new, (method)max_cv_jit_label_free, (short)sizeof(t_max_cv_jit_label), 
		0L, A_GIMME, 0);

	p = max_jit_classex_setup(calcoffset(t_max_cv_jit_label,obex));
	q = jit_class_findbyname(gensym("cv_jit_label"));    
    max_jit_classex_mop_wrap(p,q,0); 		//name/type/dim/planecount/bang/outputmatrix/etc
    max_jit_classex_standard_wrap(p,q,0); 	//getattributes/dumpout/maxjitclassaddmethods/etc
    addmess((method)max_jit_mop_assist, "assist", A_CANT,0);  //standard mop assist fn
	
	return 0;
}

void max_cv_jit_label_free(t_max_cv_jit_label *x)
{
	max_jit_mop_free(x);
	jit_object_free(max_jit_obex_jitob_get(x));
	max_jit_obex_free(x);
}

void *max_cv_jit_label_new(t_symbol *s, long argc, t_atom *argv)
{
	t_max_cv_jit_label *x,*o;

	if (x=(t_max_cv_jit_label *)max_jit_obex_new(max_cv_jit_label_class,gensym("cv_jit_label"))) {
		if (o=jit_object_new(gensym("cv_jit_label"))) {
			max_jit_mop_setup_simple(x,o,argc,argv);			
			max_jit_attr_args(x,argc,argv);
		} else {
			error("cv.jit.label: could not allocate object");
			freeobject((t_object *)x);
		}
	}
	return (x);
}