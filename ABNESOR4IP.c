/* ABNESOR.c */
#include "mex.h"
#include "blas.h"

// compressed column storage data structure
typedef struct sparseCCS
{
    double *AC;  /* numerical values, size nzmax */
    mwIndex *ia; /* row indices */
    mwIndex *jp; /* column pointers (n+1) */
    mwSize m;    /* number of rows */
    mwSize n;    /* number of columns */
} ccs;


double eps = 1.0e-6, ieps = 1.0e-1, one = 1.0, zero = 0.0;
mwSize maxnin = 50;


// How to use
void usage()
{
    mexPrintf("ABNESOR4IP: AB-GMRES method preconditioned by NE-SOR inner iterations\n");
    mexPrintf("This Matlab-MEX function is for the minimum-norm solution of \n");
    mexPrintf("linear systems Ax=b appearing in the interor point method.\n");
    mexPrintf("  x = ABNESOR4IP(A', b);\n");
    mexPrintf("  [x, relres, iter] = ABNESOR4IP(A', b, tol, maxit);\n\n");
    mexPrintf("  valuable | size | remark \n");
    mexPrintf("  A'        m-by-n   coefficient matrix. must be sparse array.\n");
    mexPrintf("  ** REMARK:   matrix A must be TRANSPOSED ** \n");
    mexPrintf("  b         n-by-1   right-hand side vector\n");
    mexPrintf("  tol       scalar   tolerance for stopping criterion.\n");
    mexPrintf("  maxit     scalar   maximum number of iterations.\n");
    mexPrintf("  x         m-by-1   resulting approximate solution.\n");
    mexPrintf("  relres   iter-by-1 relative residual history.\n");
    mexPrintf("  iter      scalar   number of iterations required for convergence.\n");
}


// Automatic parameter tuning for NE-SOR inner iterations
void opNESOR(const ccs *A, double *rhs, double *Aei, double *x, double *omg, mwIndex *nin)
{
	double *AC, d, e, res1, res10, res2 = zero, tmp, tmp1, tmp2, *r, *y, *y10;
	mwIndex i, *ia, inc1 = 1, j, *jp, k, k1, k2, l;
	mwSize m, n;

	AC = A->AC;
	ia = A->ia;
	jp = A->jp;
	m  = A->m;
	n  = A->n;

	// Allocate r
	if ((r = (double *)mxMalloc(sizeof(double) * (n))) == NULL) {
		mexErrMsgTxt("Failed to allocate r");
	}

	// Allocate y
	if ((y = (double *)mxCalloc(m, sizeof(double))) == NULL) {
		mexErrMsgTxt("Failed to allocate y");
	}

    // Allocate y10
    if ((y10 = (double *)mxCalloc(m, sizeof(double))) == NULL) {
        mexErrMsgTxt("Failed to allocate y10");
    }

	for (i=0; i<m; ++i) x[i] = zero;

	*nin = 0;

	// Tune the number of inner iterations
    for (k=0; k<maxnin; ++k) {

		for (j=0; j<n; ++j) {
			k1 = jp[j];
			k2 = jp[j+1];
			for (d=zero, l=k1; l<k2; ++l) d += AC[l]*x[ia[l]];
			d = (rhs[j] - d) * Aei[j];
			for (l=k1; l<k2; ++l) x[ia[l]] += d*AC[l];
		}

		for (d =e=zero, i=0; i<m; ++i) {
			tmp1 = fabs(x[i]);
			tmp2 = fabs(x[i] - y[i]);
			if (d < tmp1) d = tmp1;
			if (e < tmp2) e = tmp2;
		}

		if (e < ieps*d) {
			*nin = k+1;

            // w = A x
            for (j=0; j<n; ++j) {
                k1 = jp[j];
                k2 = jp[j+1];
                for (tmp=zero, l=k1; l<k2; ++l) tmp += AC[l]*x[ia[l]];
                r[j] = tmp;
            }

            for (j=0; j<n; ++j) r[j] -= rhs[j];
            res10 = dnrm2(&n, r, &inc1);

            for (i=0; i<m; ++i) y10[i] = x[i];

			break;
		}

		for (i=0; i<m; ++i) y[i] = x[i];

	}

	if (*nin == 0) {
        *nin = maxnin;

        // w = A x
        for (j=0; j<n; ++j) {
            k1 = jp[j];
            k2 = jp[j+1];
            for (tmp=zero, l=k1; l<k2; ++l) tmp += AC[l]*x[ia[l]];
            r[j] = tmp;
        }

        for (j=0; j<n; ++j) r[j] -= rhs[j];
        res10 = dnrm2(&n, r, &inc1);

        for (i=0; i<m; ++i) y10[i] = x[i];

    }

	// Tune the relaxation parameter
    *omg = 1.9;

    for (i=0; i<m; ++i) x[i] = zero;

    i = *nin;
    while (i--) {
        for (j=0; j<n; ++j) {
            k1 = jp[j];
            k2 = jp[j+1];
            for (d=zero, l=k1; l<k2; ++l) d += AC[l]*x[ia[l]];
            d = (*omg) * (rhs[j] - d) * Aei[j];
            for (l=k1; l<k2; ++l) x[ia[l]] += d*AC[l];
        }
    }

    // w = A x
    for (j=0; j<n; ++j) {
        k1 = jp[j];
        k2 = jp[j+1];
        for (tmp=zero, l=k1; l<k2; ++l) tmp += AC[l]*x[ia[l]];
        r[j] = tmp;
    }

    for (j=0; j<n; ++j) r[j] -= rhs[j];

    res2 = dnrm2(&n, r, &inc1);

    for (k=18; k>0; --k) {

        if (k != 10) {

            for (i=0; i<m; ++i) y[i] = x[i];

    		*omg = 1.0e-1 * (double)(k); // omg = 1.8, 1.8, ..., 0.1

    		for (i=0; i<m; ++i) x[i] = zero;

    		i = *nin;
    		while (i--) {
    			for (j=0; j<n; ++j) {
    				k1 = jp[j];
    				k2 = jp[j+1];
    				for (d=zero, l=k1; l<k2; ++l) d += AC[l]*x[ia[l]];
    				d = (*omg) * (rhs[j] - d) * Aei[j];
    				for (l=k1; l<k2; ++l) x[ia[l]] += d*AC[l];
    			}
    		}

    		// w = A x
    		for (j=0; j<n; ++j) {
    			k1 = jp[j];
    			k2 = jp[j+1];
    			for (tmp=zero, l=k1; l<k2; ++l) tmp += AC[l]*x[ia[l]];
    			r[j] = tmp;
    		}

    		for (j=0; j<n; ++j) r[j] -= rhs[j];

    		res1 = dnrm2(&n, r, &inc1);

    		if (res1 > res2) {
    			*omg += 1.0e-1;
    			return;
    		}

            res2 = res1;

        } else {

            if (res10 > res2) {
                *omg = 1.1e+0;
                return;
            }

            for (i=0; i<m; ++i) x[i] = y10[i];

            res2 = res10;

        }
	}

    return;

}


// Outer iterations: AB-GMRES
void ABGMRES(const ccs *A, double *b, mwIndex maxit, double *iter, double *relres, double *x){

	double *c, *g, *r, *pt, *s, *w, *y, *tmp_x, *Aei, *AC, *H, *V;
	double beta, d, inprod, min_nrmr, nrmb, nrmr, invnrmb, omg, tmp, Tol;
	mwIndex i, *ia, inc1 = 1, j, *jp, k, k1, k2, kp1, l, nin, sizeHrow = maxit+1;
	mwSize m, n;
	char charU[1] = "U", charN[1] = "N";

	AC = A->AC;
	ia = A->ia;
	jp = A->jp;
	m  = A->m;
	n  = A->n;

	// Allocate V[n * (maxit+1)]
	if ((V = (double *)mxMalloc(sizeof(double) * n * (sizeHrow))) == NULL) {
		mexErrMsgTxt("Failed to allocate H");
	}

	// // Allocate H[maxit * (maxit+1)]
	if ((H = (double *)mxMalloc(sizeof(double) * maxit * (sizeHrow))) == NULL) {
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
	if ((g = (double *)mxMalloc(sizeof(double) * (sizeHrow))) == NULL) {
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

	// Allocate tmp_x
	if ((tmp_x = (double *)mxMalloc(sizeof(double) * m)) == NULL) {
		mexErrMsgTxt("Failed to allocate tmp_x");
	}

	#define V(i, j) V[i + j*n]
	#define H(i, j) H[i + j*sizeHrow]

	iter[0] = zero;
	min_nrmr = 2.0e+52;

	for (j=0; j<n; ++j) {
		k1 = jp[j];
		k2 = jp[j+1];
		for (inprod=zero, l=k1; l<k2; ++l) inprod += AC[l]*AC[l];
		if (inprod > zero) {
			Aei[j] = one / inprod;
		} else {
			mexPrintf("%.15e\n", AC[1]);
			mexErrMsgTxt("'warning: ||aj|| = 0");
		}
	}

	// norm of b
  	nrmb = dnrm2(&n, b, &inc1);
  	invnrmb = one / nrmb;

  	// beta = ||b||2
  	beta = nrmb;

  	// Stopping criterion
  	Tol = eps * beta;

  	// beta e1
  	g[0] = beta;

  	// NE-SOR inner iterations: w = B r
   opNESOR(A, b, Aei, tmp_x, &omg, &nin);

   tmp = one / beta;
  	for (j=0; j<n; ++j) {
  		Aei[j] *= omg;
  		V[j] = tmp * b[j];	// Normalize
  	}

  	// Main loop
  	for (k=0; k<maxit; ++k) {

  		// NE-SOR inner iterations: w = B r
		for (i=0; i<m; ++i) tmp_x[i] = zero;
		i = nin;
		while (i--) {
			for (j=0; j<n; ++j) {
				k1 = jp[j];
				k2 = jp[j+1];
				for (d=zero, l=k1; l<k2; ++l) d += AC[l]*tmp_x[ia[l]];
				d = (V(j, k) - d) * Aei[j];
				for (l=k1; l<k2; ++l) tmp_x[ia[l]] += d*AC[l];
			}
		}

		// w = A x
		for (j=0; j<n; ++j) {
			k1 = jp[j];
			k2 = jp[j+1];
			for (tmp=zero, l=k1; l<k2; ++l) tmp += AC[l]*tmp_x[ia[l]];
			w[j] = tmp;
		}

		// Modified Gram-Schmidt orthogonzlization
		for (kp1=k+1, i=0; i<kp1; ++i) {
			pt = &V[i*n];
			tmp = -ddot(&n, w, &inc1, pt, &inc1);
			daxpy(&n, &tmp, pt, &inc1, w, &inc1);
			H(i, k) = -tmp;
		}

		// h_{k+1, k}
        tmp = dnrm2(&n, w, &inc1);
        H(kp1, k) = tmp;

		// Check breakdown
		if (tmp > zero) {
			for (tmp=one/tmp, j=0; j<n; ++j) V(j, kp1) = tmp * w[j];
		} else {
			mexPrintf("h_{k+1, k} = %.15e, at step %d\n", H(kp1, k), kp1);
			mexErrMsgTxt("Breakdown.");
		}

		// Apply Givens rotations

		for (i=0; i<k; ++i) {
			tmp = c[i]*H(i, k) + s[i]*H(i+1, k);
			H(i+1, k) = -s[i]*H(i, k) + c[i]*H(i+1, k);
			H(i, k) = tmp;
		}

		// Compute Givens rotations
		drotg(&H(k, k), &H(kp1, k), &c[k], &s[k]);

		// Apply Givens rotations
		tmp = -s[k] * g[k];
		nrmr = fabs(tmp);
		g[kp1] = tmp;
		g[k] = c[k] * g[k];

		relres[k] = nrmr * invnrmb;

		// mexPrintf("%d %.15e\n", k+1, relres[k]);

		if (nrmr < Tol) {

			// Derivation of the approximate solution x_k
			for (i=0; i<kp1; ++i) y[i] = g[i];

			// Backward substitution
			dtrsv(charU, charN, charN, &kp1, H, &sizeHrow, y, &inc1);

			// w = V y
			dgemv(charN, &n, &kp1, &one, &V[0], &n, y, &inc1, &zero, w, &inc1);

			// NESOR(w, x);
			for (i=0; i<m; ++i) tmp_x[i] = zero;
			i = nin;
			while (i--) {
				for (j=0; j<n; ++j) {
					k1 = jp[j];
					k2 = jp[j+1];
					for (d=zero, l=k1; l<k2; ++l) d += AC[l]*tmp_x[ia[l]];
					d = (w[j] - d) * Aei[j];
					for (l=k1; l<k2; ++l) tmp_x[ia[l]] += d*AC[l];
				}
			}

			// r = A x
			for (j=0; j<n; ++j) {
				k1 = jp[j];
				k2 = jp[j+1];
				for (tmp=zero, l=k1; l<k2; ++l) tmp += AC[l]*tmp_x[ia[l]];
				r[j] = tmp;
			}

			for (j=0; j<n; ++j) r[j] -= b[j];

			nrmr = dnrm2(&n, r, &inc1);

			if (nrmr < min_nrmr) {
				for (i=0; i<m; ++i) x[i] = tmp_x[i];
				min_nrmr = nrmr;
				iter[0] = (double)(kp1);
			}

			relres[k] = nrmr * invnrmb;

			// mexPrintf("%d, %.15e\n", k+1, nrmr/nrmb);

			// Convergence check
			if (nrmr < Tol) {

			 	mxFree(tmp_x);
		  		mxFree(y);
		  		mxFree(s);
		  		mxFree(c);
		  		mxFree(g);
		  		mxFree(Aei);
		  		mxFree(w);
		  		mxFree(r);
		  		mxFree(H);
				mxFree(V);

				// mexPrintf("Successfully converged.\n");
				// mexPrintf("nin=%d, omg=%.2e\n", nin, omg);

				return;

			}
		}
	}

	mexPrintf("Failed to converge.\n");

	// Derivation of the approximate solution x_k
	if (iter[0] == 0.0) {

		for (i=0; i<k; ++i) y[i] = g[i];

		// Backward substitution
		dtrsv(charU, charN, charN, &k, H, &sizeHrow, y, &inc1);

		// w = V y
		for (j=0; j<n; ++j) w[j] = zero;
		dgemv(charN, &n, &k, &one, &V[0], &n, y, &inc1, &zero, w, &inc1);

		// NESOR(w, x);
		for (i=0; i<m; ++i) x[i] = zero;
		i = nin;
		while (i--) {
			for (j=0; j<n; ++j) {
				k1 = jp[j];
				k2 = jp[j+1];
				for (d=zero, l=k1; l<k2; ++l) d += AC[l]*x[ia[l]];
				d = (w[j] - d) * Aei[j];
				for (l=k1; l<k2; ++l) x[ia[l]] += d*AC[l];
			}
		}

		iter[0] = (double)(k);

	}

	mxFree(tmp_x);
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


/* form sparse matrix data structure */
ccs *form_ccs(ccs *A, const mxArray *Amat)
{
    A->jp = (mwIndex *)mxGetJc(Amat);
    A->ia = (mwIndex *)mxGetIr(Amat);
    A->m = mxGetM(Amat);
    A->n = mxGetN(Amat);
    A->AC = mxGetPr(Amat);
    return (A) ;
}


// Main
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ccs *A, Amat;
    double *b, *iter, *relres, *x;
    mwIndex maxit;
    mwSize m, n;

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

    // Check the number of input arguments
    if (nrhs < 1) {
        usage();
        mexErrMsgTxt("Input A.");
    } else if (nrhs < 2) {
        usage();
        mexErrMsgTxt("Input b.");
    }

	// Check the 1st argument
    if (!mxIsSparse(prhs[0]))  {
        usage();
        mexErrMsgTxt("1st input argument must be a sparse array.");
    } else if (mxIsComplex(prhs[0])) {
        mexErrMsgTxt("1st input argument must be a real array.");
    }

    A = form_ccs(&Amat, prhs[0]);
    m = A->m;
    n = A->n;

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
   		maxit = (mwIndex)*mxGetPr(prhs[3]);
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
   ABGMRES(A, b, maxit, iter, relres, x);

   // Reshape relres
   mxSetPr(plhs[1], relres);
   mxSetM(plhs[1], (mwSize)(iter[0]));
   mxSetN(plhs[1], 1);

}
