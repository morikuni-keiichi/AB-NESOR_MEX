/* ABNESOR.c */
#include "matrix.h"
#include "mex.h"
#include <math.h>
#include <stdlib.h>
#include "blas.h"
#include <stddef.h>

double *AC, *b, *Aei;
mwIndex *ia, *jp;
double eps = 1.0e-6, omg, one = 1.0, zero = 0.0;
int nin;
size_t m, maxit, n;
// mwIndex nnz;


// How to use
void usage()
{
    mexPrintf("ABNESOR4IP: AB-GMRES method preconditioned by NE-SOR inner iterations\n");
    mexPrintf("This Matlab-MEX function is for the minimum-norm solution of \n");
    mexPrintf("linear systems Ax=b appearing in the interor point method.\n");
    mexPrintf("  x = ABNESOR4IP(A', b);\n");
    mexPrintf("  [x, y, relres, iter] = ABNESOR4IP(A', b, tol, maxit);\n\n");
    mexPrintf("  valuable | size | remark \n");
    mexPrintf("  A'         m-by-n   coefficient matrix. must be sparse array.\n");
    mexPrintf("  ** REMARK:   matrix A must be TRANSPOSED ** \n");
    mexPrintf("  b         n-by-1   right-hand side vector\n");
    mexPrintf("  tol       scalar   tolerance for stopping criterion.\n");
    mexPrintf("  maxit     scalar   maximum number of iterations.\n");
    mexPrintf("  x         m-by-1   resulting approximate solution.\n");
    mexPrintf("  y         n-by-1   resulting approximate solution.\n");
    mexPrintf("  relres   iter-by-1 relative residual history.\n");
    mexPrintf("  iter      scalar   number of iterations required for convergence.\n");
}


// Automatic parameter tuning for NE-SOR inner iterations
void opNESOR(double *rhs, double *x)
{
	double d, e, res1, res2 = zero, tmp, tmp1, tmp2, *r, *y;
	int k;
	mwSize i, j, k1, k2, l;
	ptrdiff_t inc1 = 1;

	// Allocate r
	if ((r = (double *)mxMalloc(sizeof(double) * (n))) == NULL) {
		mexErrMsgTxt("Failed to allocate r");
	}

	// Allocate y
	if ((y = (double *)mxCalloc(m, sizeof(double))) == NULL) {
		mexErrMsgTxt("Failed to allocate y");
	}

	for (i=0; i<m; i++) x[i] = y[i] = zero;

	nin = 0;

	// Tune the number of inner iterations
	k = 50;
	while(k--) {

		for (j=0; j<n; j++) {
			d = zero;
			k1 = jp[j];
			k2 = jp[j+1];
			for (l=k1; l<k2; l++) d += AC[l]*x[ia[l]];
			d = (rhs[j] - d) * Aei[j];
			for (l=k1; l<k2; l++) x[ia[l]] += d*AC[l];
		}

		d = e = zero;
		for (i=0; i<m; i++) {
			tmp1 = fabs(x[i]);
			tmp2 = fabs(x[i] - y[i]);
			if (d < tmp1) d = tmp1;
			if (e < tmp2) e = tmp2;
		}

		if (e<1.0e-1*d) {
			nin = 50 - k;
			break;
		}

		for (i=0; i<m; i++) y[i] = x[i];

	}

	if (nin == 0) nin = 50;

	// Tune the relaxation parameter
	k = 20;
	while (k--) {

		omg = 1.0e-1 * (double)(k); // omg = 1.9, 1.8, ..., 0.1

		for (i=0; i<m; i++) x[i] = zero;

		i = nin;
		while (i--) {
			for (j=0; j<n; j++) {
				d = zero;
				k1 = jp[j];
				k2 = jp[j+1];
				for (l=k1; l<k2; l++) d += AC[l]*x[ia[l]];
				d = omg * (rhs[j] - d) * Aei[j];
				for (l=k1; l<k2; l++) x[ia[l]] += d*AC[l];
			}
		}

		// w = A x
		for (j=0; j<n; j++) {
			tmp = zero;
			k1 = jp[j];
			k2 = jp[j+1];
			for (l=k1; l<k2; l++) tmp += AC[l]*x[ia[l]];
			r[j] = tmp;
		}

		for (j=0; j<n; j++) r[j] -= b[j];

		res1 = dnrm2(&n, r, &inc1);

		if (k < 19) {
			if (res1 > res2) {
				omg += 1.0e-1;
				for (i=0; i<m; i++) x[i] = y[i];
				return;
			} else if (k == 1) {
				omg = 1.0e-1;
				return;
			}
		}

		res2 = res1;

		for (i=0; i<m; i++) y[i] = x[i];
	}
}


// Outer iterations: AB-GMRES
void ABGMRES(double *iter, double *relres, double *x){

	double *V, *H, *c, *g, *r, *s, *w, *y;
	double beta, d, inprod, min_nrmr, nrmb, nrmr, tmp, Tol;
	int i, j, k;
	char charU[1] = "U", charN[1] = "N";
	mwSize k1, k2, l;
	ptrdiff_t ind_k, inc1 = 1, sizen = n, sizeHrow = maxit+1;

	#define V(i, j) V[i + j*n]
	#define H(i, j) H[i + j*(maxit+1)]

	if ((V = (double *)mxMalloc(sizeof(double) * n * (maxit+1))) == NULL) {
		mexErrMsgTxt("Failed to allocate H");
	}

	// // Allocate H[maxit * (maxit+1)]
	if ((H = (double *)mxMalloc(sizeof(double) * maxit * (maxit+1))) == NULL) {
		mexErrMsgTxt("Failed to allocate H");
	}

	// Allocate r
	if ((r = (double *)mxMalloc(sizeof(double) * n)) == NULL) {
		mexErrMsgTxt("Failed to allocate r");
	}

	// Allocate w
	if ((w = (double *)mxMalloc(sizeof(double) * n)) == NULL) {
		mexErrMsgTxt("Failed to allocate w");
	}

	// Allocate Aei
	if ((Aei = (double *)mxMalloc(sizeof(double) * n)) == NULL) {
		mexErrMsgTxt("Failed to allocate Aei");
	}

	// Allocate g
	if ((g = (double *)mxMalloc(sizeof(double) * (maxit+1))) == NULL) {
		mexErrMsgTxt("Failed to allocate g");
	}

	// Allocate c
	if ((c = (double *)mxMalloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate c");
	}

	// Allocate s
	if ((s = (double *)mxMalloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate s");
	}

	// Allocate y
	if ((y = (double *)mxMalloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate y");
	}

	iter[0] = zero;
	min_nrmr = 2.0e+52;

	for (j=0; j<n; j++) {
		inprod = zero;
		k1 = jp[j];
		k2 = jp[j+1];
		for (l=k1; l<k2; l++) inprod += AC[l]*AC[l];
		if (inprod > zero) {
			Aei[j] = one / inprod;
		} else {
			mexPrintf("%.15e\n", AC[1]);
			mexErrMsgTxt("'warning: ||aj|| = 0");
		}
	}

	// norm of b
  	nrmb = dnrm2(&n, b, &inc1);

  	// beta = ||b||2
  	beta = nrmb;

  	// Stopping criterion
  	Tol = eps * beta;

  	// beta e1
  	g[0] = beta;

  	// NE-SOR inner iterations: w = B r
  	opNESOR(b, x);

  	// mexErrMsgTxt("Stop");

	tmp = one / beta;
  	for (j=0; j<n; j++) {
  		Aei[j] *= omg;
  		V(j, 0) = tmp * b[j];	// Normalize
  	}

  	// Main loop
  	for (k=0; k<maxit; k++) {

  		// NE-SOR inner iterations: w = B r
		for (i=0; i<m; i++) x[i] = zero;
		i = nin;
		while (i--) {
			for (j=0; j<n; j++) {
				d = zero;
				k1 = jp[j];
				k2 = jp[j+1];
				for (l=k1; l<k2; l++) d += AC[l]*x[ia[l]];
				d = (V(j, k) - d) * Aei[j];
				for (l=k1; l<k2; l++) x[ia[l]] += d*AC[l];
			}
		}

		// w = A x
		for (j=0; j<n; j++) {
			tmp = zero;
			k1 = jp[j];
			k2 = jp[j+1];
			for (l=k1; l<k2; l++) tmp += AC[l]*x[ia[l]];
			w[j] = tmp;
		}

		// Modified Gram-Schmidt orthogonzlization
		for (i=0; i<k+1; i++) {
			tmp = -ddot(&n, w, &inc1, &V[i*n], &inc1);
			daxpy(&n, &tmp, &V[i*n], &inc1, w, &inc1);
			H(i, k) = -tmp;
		}

		// h_{k+1, k}
		tmp = dnrm2(&n, w, &inc1);
		H(k+1, k) = tmp;

		// Check breakdown
		if (tmp > zero) {
			tmp = one / tmp;
			for (j=0; j<n; j++) V(j, (k+1)) = tmp * w[j];
		} else {
			mexPrintf("h_k+1, k = %.15e, at step %d\n", H(k+1, k), k+1);
			mexErrMsgTxt("Breakdown.");
		}

		// Apply Givens rotations

		for (i=0; i<k; i++) {
			tmp = c[i]*H(i, k) + s[i]*H(i+1, k);
			H(i+1, k) = -s[i]*H(i, k) + c[i]*H(i+1, k);
			H(i, k) = tmp;
		}

		tmp = H(k, k);

		// Compute Givens rotations
		drotg(&tmp, &H(k+1, k), &c[k], &s[k]);

		H(k, k) = tmp;

		// Apply Givens rotations
		g[k+1] = -s[k] * g[k];
		g[k] = c[k] * g[k];

		nrmr = fabs(g[k+1]);

		relres[k] = nrmr / nrmb;

		// mexPrintf("%d %.15e\n", k+1, relres[k]);

		if (nrmr < Tol) {

			// Derivation of the approximate solution x_k
			ind_k = k+1;
			for (l=0; l<ind_k; l++) y[l] = g[l];

			// Backward substitution
			dtrsv(charU, charN, charN, &ind_k, H, &sizeHrow, y, &inc1);

			// w = V y
			dgemv(charN, &n, &ind_k, &one, &V[0], &sizen, y, &inc1, &zero, w, &inc1);

			// NESOR(w, x);
			for (i=0; i<m; i++) x[i] = zero;
			i = nin;
			while (i--) {
				for (j=0; j<n; j++) {
					d = zero;
					k1 = jp[j];
					k2 = jp[j+1];
					for (l=k1; l<k2; l++) d += AC[l]*x[ia[l]];
					d = (w[j] - d) * Aei[j];
					for (l=k1; l<k2; l++) x[ia[l]] += d*AC[l];
				}
			}

		 	iter[0] = (double)(k+1);

	  		mxFree(y);
	  		mxFree(s);
	  		mxFree(c);
	  		mxFree(g);
	  		mxFree(Aei);
	  		mxFree(w);
	  		mxFree(r);
	  		mxFree(H);
			mxFree(V);

			// mexPrintf("Required number of iterations: %d\n", (int)(*iter));
			// mexPrintf("Successfully converged.\n");
			// mexPrintf("nin=%d, omg=%.2e\n", nin, omg);

			return;

		}

	}

	mexPrintf("Failed to converge.\n");

	// Derivation of the approximate solution x_k
	if (iter[0] == 0.0) {

		ind_k = k;
		for (l=0; l<ind_k; l++) y[l] = g[l];

		// Backward substitution
		dtrsv(charU, charN, charN, &ind_k, H, &sizeHrow, y, &inc1);

		// w = V y
		for (j=0; j<n; j++) w[j] = zero;
		dgemv(charN, &n, &ind_k, &one, &V[0], &sizen, y, &inc1, &zero, w, &inc1);

		// NESOR(w, x);
		for (i=0; i<m; i++) x[i] = zero;
		i = nin;
		while (i--) {
			for (j=0; j<n; j++) {
				d = zero;
				k1 = jp[j];
				k2 = jp[j+1];
				for (l=k1; l<k2; l++) d += AC[l]*x[ia[l]];
				d = (w[j] - d) * Aei[j];
				for (l=k1; l<k2; l++) x[ia[l]] += d*AC[l];
			}
		}

		iter[0] = (double)(k);

	}

	mxFree(y);
	mxFree(s);
	mxFree(c);
	mxFree(g);
	mxFree(Aei);
	mxFree(w);
	mxFree(r);
	mxFree(H);
	mxFree(V);

	return;

}


// Main
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *iter, *relres, *x, *y;
	// mwSize nzmax;

	// Check the number of input arguments
    if(nrhs > 4){
    	usage();
        mexWarnMsgTxt("Too many inputs. Ignored extras.\n");
    }

    // Check the number of output arguments
    if(nlhs > 4){
        mexWarnMsgTxt("Too many outputs. Ignored extras.");
    }

    /* Check for proper number of input and output arguments */
	if (nrhs < 2) {
		usage();
        mexErrMsgTxt("Please input b.");
    }

    // Check the number of input arguments
    if (nrhs < 1) {
    	usage();
        mexErrMsgTxt("Please input A.");
    }

	// Check the 1st argument
    if (!mxIsSparse(prhs[0]))  {
		usage();
        mexErrMsgTxt("1st input argument must be a sparse array.");
    }

    if (mxIsComplex(prhs[0])) {
    	mexErrMsgTxt("1st input argument must be a real array.");
    }

    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    // nnz = *((int)mxGetJc(prhs[0]) + n);
    // nzmax = mxGetNzmax(prhs[0]);

    ia = mxGetIr(prhs[0]);
    jp = mxGetJc(prhs[0]);
    AC = mxGetPr(prhs[0]);

	// Check the 2nd argument
    if (mxGetM(prhs[1]) != n) {
    	usage();
    	mexErrMsgTxt("The length of b is not the numer of columns of A'.");
    }

    b = mxGetPr(prhs[1]);

	// Check the 3rd argument
    // Set eps
    if (nrhs < 3) {
        mexPrintf("Default: stopping criterion is set to 1e-6.\n");
    } else {
    	if (mxIsComplex(prhs[2]) || mxGetM(prhs[2])*mxGetN(prhs[2])!=1) {
    		usage();
    		mexErrMsgTxt("3nd argument must be a scalar");
    	} else {
    		eps = *(double *)mxGetPr(prhs[2]);
    		if (eps<zero || eps>=one) {
    			usage();
    			mexErrMsgTxt("3nd argument should be positive and less than or equal to 1.");
    		}
    	}
    }

	// Check the 4th argument
	// Set maxit
    if (nrhs < 4) {
    	maxit = n;
    	// mexPrintf("Default: max number of iterations is set to the number of columns.\n");
   	} else {
   		if (mxIsComplex(prhs[3]) || mxGetM(prhs[3])*mxGetN(prhs[3])!=1) {
   			usage();
    		mexErrMsgTxt("4th argument must be a scalar");
    	} else {
   			maxit = (size_t)*mxGetPr(prhs[3]);
   			if (maxit < 1) {
   				usage();
   				mexErrMsgTxt("4th argument must be a positive scalar");
   			}
   		}
	}

	plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(maxit, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);

    x = mxGetPr(plhs[0]);
    relres = mxGetPr(plhs[1]);
    iter = mxGetPr(plhs[2]);

	// AB-GMRES method for IP
    ABGMRES(iter, relres, x);

    // Reshape relres
    mxSetPr(plhs[1], relres);
    mxSetM(plhs[1], (int)*iter);
    mxSetN(plhs[1], 1);

}
