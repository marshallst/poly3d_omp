/*==================================================
SET TABSTOPS AT EVERY FOUR SPACES FOR PROPER DISPLAY
====================================================*/

/*****************************************************************************
* FILE: poly3d.c, version 0.0 (beta)
* DATE: June, 1993
* BY:   Andrew L. Thomas
*
* A 3-D polygonal boundary element code based on superposition of angular
* dislocations (Yoffe, 1956; Comninou & Dunders, 1966).  Translated and
* expanded from a triangular element FORTRAN code written by M. Jeyakumaran
* at Northwestern University (Jeyakumaran et al., 1992).
*
* See the "Poly3D Users Manual" for more information.
*****************************************************************************/


/**************************** Revision History *******************************
*
* If you change this code in ANY way, describe and initial those changes
* below.  Put a comment line containing the word CHANGE (all uppercase)
* where the changes are made, so they will be easy to find in the source code.
*
* DATE		FILE			NAME			      
* --------	------------	----------------------
* Jun-1993	poly3d.c		Andrew L. Thomas
*	Version 0.0 (beta) of Poly3D completed
* Nov-1997	poly3d.c		Yann Lagalaye
*	Version 1.1 Correction of "shadow effect"
* Sept-2014	poly3d.c		Larry Baker (USGS)
*	Version 1.2 Corrected memory overflow issues that are no longer allowed in C code
* Jan-2015	poly3d.c		Scott T. Marshall
*	Version 1.2 Modified several printf/fprintf statements to stdout
* Mar-2015	poly3d.c		Scott T. Marshall
*	Version 1.2 Modified several printf/fprintf statements to make output file data line up nicely
* Jan-2026	poly3d.c		Scott T. Marshall
*	Version 3.0: Compiles as poly3d_omp. Renamed to poly3d_omp to reflect the OpenMP implementation.
*	Skipped to version 3.0 to avoid confusion with old 2.x versions previously maintained by IGEOSS.
*	Major updates:
*	  Code is now parallelized! This parallelization was accomplished with the assistance of Google Gemini CLI.
*     Parallelized steps: building IC matrix, solving IC matrix, calculating/writing element data and Obs pts
*	  Uses OpenMP for loops and LAPACK/OpenBlas for the IC matrix solver.
*     To accomplish this, the function, determine_burgers_vectors, has been significantly re-written.
*	  The IC matrix is now solved with LAPACK (dgetrf, dgetrs).
*   Minor updates:
*     Many minor code fixes were made to address modern compiler warnings/errors.
*     Removed all K&R style function declarations as I am sure no one uses this ancient compiler any longer.
*     Fixed output when no command args are given and when -h (help) is given. Now exits properly.
*	  Feedback to STDOUT is improved to show user realtime model progress and times for each step.
*	  Condition numbers are printed in exponential format if 1e5 or larger.
*****************************************************************************/

/******************************** Includes **********************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h> 
#include <time.h>
#include <assert.h>
#include "matrix.h"
#include "safetan.h"
#include "getoptPoly3D.h"
#include "getwords.h"
#include "infcoeff.h"
#include "elastic.h"
#include "pi.h"
#include "nrutil.h"
#include "nr.h"
/* Standard OpenMP Include (for parallelized code)*/
#ifdef _OPENMP
#include <omp.h>
#endif


/******************************** Constants *********************************/
/*-------------
 MISCELLANEOUS
--------------*/
#define MAXFILE	 			256			/* Max length of file names			*/
#define MAX_ERROR_MSG		512			/* Max length of error messages		*/
#define MAXLINE				256			/* Max line length for getwords()	*/
#define MAXWORDS			50			/* Max # words on getwords() line	*/
#define GLOBAL_NAME			"global"	/* Global coord system name			*/
#define ELT_CSYS_NAME		"elocal"	/* Element coord system name		*/
#define END_STMT			"end"		/* End flag for input file sections	*/
#define COMMENT_CHAR		'*'			/* Input file comment character		*/
#define CONTINUE_CHAR		'\\'		/* Input file line continuation char*/
#define ERROR				-1			/* Return value	for function errors	*/
#define FALSE				0			/* False flag						*/
#define TRUE				1			/* True flag						*/
#define BVECTOR_BC			0			/* Burgers vector BC component flag	*/
#define TRACTION_BC			1			/* Traction BC component flag		*/

/*-----------
 PROGRAM INFO
-------------*/
#define PROGRAM			"poly3d_omp"		
#define VERSION			"3.0"
#ifdef __DATE__						
#define COMPILE_DATE	__DATE__		
#else
#define COMPILE_DATE "Date Unavailable"
#endif

/*--------------------------------------
 NUMERICAL LIMITS USED WHEN CHECKING...
---------------------------------------*/
#define SWAP_TINY			1.0e-10		  /* ...if vertices must be swapped	*/
/*Changed by Scott T. Marshall, to work with kinematic models, where long skinny elements are fine */
/*#define TINY_ANGLE			1.0e-16*/ /* ...if elt coord sys can be calc */
#define TINY_ANGLE			0.5*PI/180.	  /* ...if elt coord sys can be calc */
#define BVERT_TINY			1.0e-10		  /* ...if point lies below a vertex */
#define COPLANAR_LIMIT		30.			  /* ...if elt vertices are co-planar */

/*-------------------------------------
 PRINT OPTION ARRAY SIZE AND POSITIONS
--------------------------------------*/
#define NUM_PR_OPTS			5
#define DISPL				0
#define	STRAIN				1
#define PSTRAIN				2
#define STRESS				3
#define PSTRESS				4

/*--------------------------------------------------------------
 CHARS USED IN INPUT FILE PRINT STRINGS TO ENABLE PRINT OPTIONS
---------------------------------------------------------------*/
#define DISPL_CHAR			'd'
#define STRESS_CHAR			's'
#define STRAIN_CHAR			'e'
#define TRACTION_CHAR		't'
#define BVECTOR_CHAR		'b'
#define PRINCIPAL_CHAR		'p'

/*----------------------------------------
 INPUT FILE FORMAT FOR DEFINING CONSTANTS
-----------------------------------------*/
#define CONST_NAME_POS		0
#define CONST_VALUE_POS		2
#define CONST_NUM_PARAMS	3

/*------------------------------------------------------
 INPUT FILE FORMAT FOR DEFINING USER COORDINATE SYSTEMS
-------------------------------------------------------*/
#define CS_NAME_POS			0
#define CS_PARENT_POS		1
#define CS_ORIGIN_POS		2
#define CS_ROT_POS			5
#define CS_ROT_ORDER_POS	8
#define CS_NUM_PARAMS		9

/*------------------------------------------------
 INPUT FILE FORMAT FOR DEFINING OBSERVATION GRIDS
-------------------------------------------------*/
#define OG_NAME_POS			0
#define OG_DIMEN_POS		1
#define OG_PRINT_OPS_POS	2
#define OG_INPUT_CSYS_POS	3
#define OG_OBSPT_CSYS_POS	4
#define OG_DATA_CSYS_POS	5
#define OG_BEGIN_POS		6
#define OG_END_POS			9
#define OG_NUMPTS_POS		12
#define OG_MIN_NUM_PARAMS	9

/*---------------------------------------
 INPUT FILE FORMAT FOR DEFINING VERTICES
----------------------------------------*/
#define V_CHAR				'v'
#define V_CHAR_POS			0
#define V_NAME_POS			1
#define V_CSYS_POS			2
#define V_X_POS				3
#define V_NUM_PARAMS		6

/*--------------------------------------
 INPUT FILE FORMAT FOR DEFINING OBJECTS
---------------------------------------*/
#define OBJ_CHAR			'o'
#define OBJ_CHAR_POS		0
#define OBJ_NAME_POS		1
#define	OBJ_PRINT_OPS_POS	2
#define OBJ_POS_CSYS_POS	3
#define OBJ_MIN_NUM_PARAMS	2

/*---------------------------------------
 INPUT FILE FORMAT FOR DEFINING ELEMENTS
----------------------------------------*/
#define E_CHAR				'e'
#define E_CHAR_POS			0
#define E_NUM_VERT_POS		1
#define E_BC_CSYS_POS		2
#define E_BC_TYPE_POS		3
#define E_BC_POS			4
#define E_VERTEX_POS		7
#define E_MIN_NUM_PARAMS	10

/*------------------------------------
 FORMAT FOR PRINTING ELEMENT GEOMETRY
-------------------------------------*/
#define ELT_GEOM_LABELS \
	" ELT Vertex Name                  X1         X2         X3\n"
#define ELT_GEOM_UNDLNS \
	"---- -------------------- ---------- ---------- ----------\n"
#define ELT_GEOM_FMT \
	"%4d %-20s %10.4f %10.4f %10.4f\n"


/*-----------------------------------------
 FORMAT FOR PRINTING OBSERVATION GRID DATA
------------------------------------------*/
#define OG_LOC_LABELS \
	"                X1                 X2                 X3 "
#define OG_LOC_UNDLNS \
	"------------------ ------------------ ------------------ "
#define OG_LOC_FMT \
	"%18.7f %18.7f %18.7f "
#define OG_DISPL_TITLE \
	"\nDISPLACEMENTS:\n\n"
#define OG_DISPL_LABELS \
	"                U1                 U2                 U3\n"
#define OG_DISPL_UNDLNS \
	"------------------ ------------------ ------------------\n"
#define OG_DISPL_FMT \
	"%18.10e %18.10e %18.10e\n"
#define OG_STRAIN_TITLE \
	"\nSTRAINS:\n\n"
#define OG_STRAIN_LABELS \
	"               E11                E22                E33                E12                E23                E13\n"
#define OG_STRAIN_UNDLNS \
	"------------------ ------------------ ------------------ ------------------ ------------------ ------------------\n"
#define OG_STRAIN_FMT \
	"%18.10e %18.10e %18.10e %18.10e %18.10e %18.10e\n"
#define OG_STRESS_TITLE \
	"\nSTRESSES:\n\n"
#define OG_STRESS_LABELS \
	"     SIG11      SIG22      SIG33      SIG12      SIG23      SIG13\n"
#define OG_STRESS_UNDLNS \
	"------------------ ------------------ ------------------ ------------------ ------------------ ------------------\n"
#define OG_STRESS_FMT \
	"%18.10e %18.10e %18.10e %18.10e %18.10e %18.10e\n"
#define OG_PSTRAIN_TITLE \
	"\nPRINCIPAL STRAINS:\n\n"
#define OG_PSTRAIN_LABELS \
	"       N1        N2        N3                 E1        N1        N2        N3                 E2        N1        N2        N3                 E3\n"
#define OG_PSTRAIN_UNDLNS \
	"--------- --------- --------- ------------------ --------- --------- --------- ------------------ --------- --------- --------- ------------------\n"
#define OG_PSTRAIN_FMT \
	"%9.5f %9.5f %9.5f %18.10e %9.5f %9.5f %9.5f %18.10e %9.5f %9.5f %9.5f %18.10e\n"
#define OG_PSTRESS_TITLE \
	"\nPRINCIPAL STRESSES:\n\n"
#define OG_PSTRESS_LABELS \
	"       N1        N2        N3               SIG1        N1        N2        N3               SIG2        N1        N2        N3               SIG3\n"
#define OG_PSTRESS_UNDLNS \
	"--------- --------- --------- ------------------ --------- --------- --------- ------------------ --------- --------- --------- ------------------\n"
#define OG_PSTRESS_FMT \
	"%9.5f %9.5f %9.5f %18.10e %9.5f %9.5f %9.5f %18.10e %9.5f %9.5f %9.5f %18.10e\n"


/*-------------------------------
 FORMAT FOR PRINTING OBJECT DATA
--------------------------------*/
#define OBJ_LOC_LABELS \
	"   ELT                X1C                X2C                X3C "
#define OBJ_LOC_UNDLNS \
	"------ ------------------ ------------------ ------------------ "
#define OBJ_LOC_FMT \
	"%6d %18.7f %18.7f %18.7f "
#define OBJ_DISPL_TITLE \
	"\nDISPLACEMENTS:\n\n"
#define OBJ_DISPL_LABELS \
	"                B1              U1(+)              U1(-)                 B2              U2(+)              U2(-)                 B3              U3(+)              U3(-) " 
#define OBJ_DISPL_UNDLNS \
	"------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ "
#define OBJ_DISPL_FMT \
	"%18.10e %18.10e %18.10e %18.10e %18.10e %18.10e %18.10e %18.10e %18.10e "
#define OBJ_STRESS_TITLE \
	"\nSTRESSES (TRACTIONS):\n\n"
#define OBJ_STRESS_LABELS \
	"                T1                 T2                 T3 "
#define OBJ_STRESS_UNDLNS \
	"------------------ ------------------ ------------------ "
#define OBJ_STRESS_FMT \
	"%18.10e %18.10e %18.10e "
#define OBJ_BC_CSYS_LABELS \
	"Coord Sys\n"
#define OBJ_BC_CSYS_UNDLNS \
	"---------\n"
#define OBJ_BC_CSYS_FMT \
	"%s\n"


/********************************** Macros **********************************/
#define RADIANS(A) ((A)*PI/180.)		/* Convert degrees to radians		*/
#define MAX(A,B) (((A) > (B)) ? (A):(B))/* MAX macro						*/


/******************************** Structures ********************************/
struct csys_s {							/* -- COORDINATE SYSTEM STRUCTURE -	*/
	char			*name;				/* Coordinate system name			*/
	double			origin[3];			/* Coord sys origin	(global)		*/
	double			local_rot[3][3];	/* (To) global rotation matrix		*/
	struct csys_s	*next;				/* Ptr to next c.s. in linked list	*/
};
typedef struct csys_s csys_t;

struct obs_grid_s {						/* -- OBSERVATION GRID STRUCTURE --	*/
	char			*name;				/* Observation grid name			*/
	int				dimension;			/* Dimension of observation grid	*/
	double			begin[3];			/* Obs grid beginning coords		*/
	double			end[3];				/* Obs grid ending coords			*/
	int				numpts[3];			/* No of obs pts along x1,x2,x3		*/
	int				print[NUM_PR_OPTS];	/* Print options array				*/
	csys_t			*endpt_csys;		/* Input coordinate system			*/
	csys_t			*obspt_csys;		/* Observation grid coord system	*/
	csys_t			*outp_csys;			/* Output coord sys for obs grid	*/
	struct obs_grid_s *next;			/* Ptr to next o.l. in linked list	*/
};
typedef struct obs_grid_s obs_grid_t;

struct vert_s {							/* ------- VERTEX STRUCTURE -------	*/
	char			*name;				/* Vertex name						*/
	double			x[3];				/* Vertex coordinates (global)		*/
	csys_t			*csys;				/* Coordinate system for vertex		*/
	struct vert_s	*next;				/* Ptr to next vertex in linked list*/
};
typedef struct vert_s vert_t;

struct disloc_seg_s {					/* --- DISLOC SEGMENT STRUCTURE ---	*/
	double 			elt_b[3][3];		/* Proj of element b to segment b	*/
	double			trend;				/* Strike of plunging leg of d.s.	*/
	double			plunge;				/* Plunge of plunging leg of d.s.	*/
	double			local_rot[3][3];	/* Local-to-global rotation matrix	*/
	vert_t			*vert[2];			/* Dislocation segment vertices		*/
};
typedef struct disloc_seg_s disloc_seg_t;

struct elt_s {							/* ------ ELEMENT STRUCTURE -------	*/
	int				num_vertices;		/* Number of vertices				*/
	int				bc_type[3];			/* Boundary condition type array	*/
	double			bc[3];				/* Boundary condition magnitudes	*/
	csys_t			elt_csys;			/* Element-local coordinate system	*/
	double			*b[3];				/* Burgers vector array				*/
	disloc_seg_t	 *disloc_seg;		/* Dislocation segment array		*/
	csys_t			*bc_csys;			/* Ptr to coord sys for element BCs	*/
	struct elt_s	*next;				/* Ptr to next elt in linked list	*/
};
typedef struct elt_s elt_t;

struct obj_s {							/* ------ OBJECT STRUCTURE -------- */
	char			*name;				/* Object type						*/
	int				print[NUM_PR_OPTS];	/* Print options					*/
	csys_t			*pos_csys;			/* Position coordinate system		*/
	elt_t			*first_elt;			/* Pointer to first element			*/
	elt_t			*last_elt;			/* Pointer to last element			*/
	struct obj_s	*next;				/* Ptr to next obj in linked list	*/
};
typedef struct obj_s obj_t;


/*************************** External Variables *****************************/
char		*title1_E = NULL;			/* Problem title					*/
char		*title2_E = NULL;			/* Problem subtitle					*/
int			half_space_E = TRUE;		/* Half/whole space flag			*/
int			check_cond_num_E = TRUE;	/* Check matrix condition num flag	*/
double		cond_num_E = -1.0;			/* Matrix condition number			*/
char		infile_E[MAXFILE];			/* Input file name					*/
char		outfile_E[MAXFILE];			/* Output file name					*/
int			linenum_E = 0;				/* Current line # in input file		*/
int			num_elts_E = 0;				/* Number of elements				*/
int			below_vertex_E = FALSE;		/* Below vertex flag				*/
double		null_value_E = -999.0;		/* Null output value				*/

/*********************************************************************************************************/
/************************* NEW: added 98-12-09 to reflect the problem of observation point near vertices */
double		coef_exclu_E   = 0.0;		/* coef exclusion value */
int			near_vertex_E = FALSE;		/* nearnest vertex flag for observation point */
/*       EXPLANATIONS: 
    Let's obs_pt be an observation point.
    Put flag near_vertex_E to FALSE.
    For each vertex in current project:
    Let's d be the mean length of segments containing v
    Let's l be the distance from v to a choosen observation point
    Then if l<d*coef_exclu => near_vertex_E = TRUE
    
    Then, before printing computed values for this observation point:
      if near_vertex_E=TRUE => print null_value_E
      otherwise print computed values
      
    So coef_exclu_E = 1.0, means 100% of the mean length of segments containing v.
*/

/*********************************************************************************************************/
/* NEW: added 2025-12-19 by STM to work with the parallelized LAPACK solver */
/*********************************************************************************************************/
extern void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern void dgecon_(char *norm, int *n, double *a, int *lda, double *anorm, double *rcond, double *work, int *iwork, int *info);
extern void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
/* NEW: added 2025-12-19 to allow for parallel writing of output files */
#pragma omp threadprivate(near_vertex_E)


/*********************************************************************************************************/

FILE		*ifp_E	= NULL;			/* Input file ptr  (default stdin)  */
FILE		*ofp_E	= NULL;			/* Output file ptr (default stdout) */
FILE		*tempfp_E[NUM_PR_OPTS];		/* Temporary file ptrs				*/

double		shear_mod_E		= -1.0;		/* Shear modulus					*/
double		psn_ratio_E		= -1.0;		/* Poisson's ratio					*/
double		youngs_mod_E	= -1.0;		/* Young's modulus					*/
double		bulk_mod_E		= -1.0;		/* Bulk modulus						*/
double		lame_lambda_E	= -1.0;		/* Lame's lambda					*/

int			print_elt_geom_E = FALSE;	/* Print element geometry flag		*/
char		*elt_geom_csys_name_E =NULL;/* Element geometry coord sys name	*/

int			rem_stress_bc_E = TRUE;		/* Remote stress vs strain bc flag	*/
double		rem_stress_E[3][3];			/* Remote stress tensor				*/
double		rem_strain_E[3][3];			/* Remote strain tensor				*/

double		*b_vector_E;				/* Burger's vector array			*/
double		**ic_matrix_E;				/* Influence coeff matrix			*/

csys_t		*first_csys_E     = NULL;	/* 1st memb of csys linked list		*/
obs_grid_t	*first_obs_grid_E = NULL;	/* 1st memb of obs grid linked list	*/
obj_t		*first_obj_E      = NULL;	/* 1st memb of obj linked list		*/
elt_t		*first_elt_E      = NULL;	/* 1st memb of elt linked list		*/
vert_t		*first_vert_E     = NULL;	/* 1st memb of vert linked list	*/


/************************* ANSI Function Declarations ***********************/
#if defined(__STDC__) || defined(ANSI) /* ANSI */
double	array_max_norm(double **a, int start_row, int end_row, int start_col,
		int	end_col);
int		calc_elt_parameters(elt_t *current_elt);
void	close_temp_files(void);
void	copy_temp_files(void);
void	determine_burgers_vectors(void);
void	displ_strain(int calc_displ, int calc_strain, double x[3],
		double displ[3], double strain[3][3], elt_t *omit_elt);
void	displ_strain_poly_elt(int calc_displ, int calc_strain, 
		elt_t *current_elt, double x[3], double displ[3],
		double strain[3][3], elt_t *omit_elt, int under);
void	print_obj_data(void);
csys_t	*find_csys(const char *name);
vert_t	*find_vert(const char *name);
int		get_double_var(double *var, const char *var_name, const char *const word[], int numwords);
int		get_boolean_var(int *var, const char *var_name, const char *true_string,
		const char *false_string, const char *const word[], int numwords, const char *line);
int		get_text_var(char **var, const char *var_name, const char *const word[],
		int numwords, const char *line);
void	get_elt_info(elt_t **current_elt, obj_t *current_obj, int numwords,
		char *word[], char *line);
void	get_obj_info(obj_t **current_obj, int numwords, char *word[],
		char *line);
void	get_program_args(void);
void	get_vert_info(vert_t **current_vert, int numwords, char *word[],
		char *line);
void	displ_strain_ics_poly_elt(int calc_displ, int calc_strain, 
		elt_t *current_elt, double x[3],
		double displ_ic[3][3], double strain_ic[3][3][3],
		elt_t *omit_elt);
void	print_obs_grid_data(void);
int		open_files();
void	open_temp_files(int print[]);
int		parse_command_line_args(int argc, char *argv[]);
/*void	p_error(const char *error_msg, const char *line);*/
void	display_msg(char *_msg);
void	print_elt_data(obj_t *current_obj,
		elt_t *current_elt, int elt_num, double displ[3],
		double  stress[3][3]);
void	print_elt_geometry(void);
void	print_obj_titles(obj_t *current_obj);
void	print_obs_grid_titles(obs_grid_t *current_obs_grid);
void	print_obs_pt_data(obs_grid_t *current_obs_grid, double x[3],
		double displ[3], double strain[3][3]);
void	print_problem_info(void);
int		read_csystems();
int		read_objs_elts_verts();
void	read_infile(void);
int		read_line(char *line, char *word[]);
int		read_observation_grids();
int		read_constants();
void	setup_global_coords(void);

/* Numerical Recipes Utils - Explicitly defined to prevent implicit int return */
/*double **dmatrix(long nrl, long nrh, long ncl, long nch);
double *dvector(long nl, long nh);*/
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);
void free_dvector(double *v, long nl, long nh);
/* Poly3D Utils */
void rotate_vector(int mode, double rotation[3][3], double vector[3]);
/* Error Handler (Matches your definition with const char*) */
void p_error(const char *error_msg, const char *line);
#endif


/******************************* Function: main ******************************
* In:	argc	- number of command line arguments
*		argv	- array of command line arguments
*****************************************************************************/
int main(int argc, char *argv[])
{
	time_t   start, t_reading, t_matrix, t_problem, t_object, t_obs,finish; 
	double   elapsed_time;

	ifp_E  = stdin;			/* Input file ptr  (default stdin)  */
	ofp_E  = stdout;		/* Output file ptr (default stdout) */

	/* Use get_program_args() or parse_command_line_args() to get
	   file names and program options.
	-------------------------------------------------------------*/
	/* 2025-12 STM: removed FPROMPT interactive mode here. See original code if you want to restore this option.*/
	if (parse_command_line_args(argc,argv) == ERROR) exit(0);
 
	/* Start counter */ 
	time( &start ); 

	/* Open the input and output files
	----------------------------------*/
	open_files();

	/* Read input file and set up the problem
	-----------------------------------------*/
	read_infile(); 
 
	time( &t_reading ); 
	elapsed_time = difftime( t_reading, start ); 
	/* Commented this line out since most input files will be read in less than a second
	Changed by Scott T. Marshall 2015-10-18
	printf( "\nReading input file :  %.0f seconds", elapsed_time );
	The rest of the printf commands here were significantly modified by Scott T. Marshall 2026-01-27
	The code now provides essentially a status update showing the user what stage it is at during a model run.
	-----------------------------------------------------------------------------------*/
	printf( "\n-------------------------------------------------------------\n" );
	printf( "Input file : %s\n", infile_E );
	printf( "-------------------------------------------------------------\n" );
	/* Solve for burger's vector for each element
	---------------------------------------------*/ 
	determine_burgers_vectors(); 
 	time( &t_matrix ); 
	elapsed_time = difftime( t_matrix,t_reading ); 
	printf( "IC matrix inversion  : %.0f seconds\n", elapsed_time );
	printf( "Condition number     : " );
	if (cond_num_E < 0.0)  printf( "(no traction bc's -> no matrix needed)\n" );
	else {
		if (check_cond_num_E){
			if (cond_num_E < 1e5)  printf( "%.2f\n", cond_num_E );
			else                   printf( "%.4e\n", cond_num_E );
		}
		else  printf( "(not requested)\n" );
	}
		
	/* Print the model setup information (header of outfile)
	-------------------------*/
	printf( "-------------------------------------------------------------\n" );
	printf( "Output file : %s\n", outfile_E );
	printf( "-------------------------------------------------------------\n" );
	printf( "Printing model setup  : " ); fflush(stdout);
	print_problem_info(); 
	time( &t_problem ); 
	elapsed_time = difftime( t_problem,t_matrix ); 
	printf( "%.0f seconds\n", elapsed_time );

	/* Calculate Burger's vectors, disps, and tractions on elements
	---------------------------------------------*/ 
	printf( "Printing element data : "); fflush(stdout);
	print_obj_data(); 
 	time( &t_object ); 
	elapsed_time = difftime( t_object,t_problem ); 
	printf( "%.0f seconds\n", elapsed_time ); 
	
	/* Calculate displacements and stresses at observation points/lines/grids
	---------------------------------------------------------------*/ 
	printf( "Printing obs pt data  : "); fflush(stdout);
	print_obs_grid_data(); 
 	time( &t_obs ); 
	elapsed_time = difftime( t_obs,t_object ); 
	printf( "%.0f seconds\n", elapsed_time ); 
 
	/* End counter, and print elapsed time */ 
	time( &finish ); 
	elapsed_time = difftime( finish, start ); 
	printf( "-------------------------------------------------------------\n" );
	printf( "Model Summary Info\n" );
	printf( "-------------------------------------------------------------\n" );
	printf( "Input file    : %s\n", infile_E );
	printf( "Output file   : %s\n", outfile_E ); 
	
	/* Print condition number of the influence coefficient matrix to stdout
	   Added by Scott T. Marshall 2015-10-18
	-------------------------------------------------------------*/
	printf( "Condition num : " );
	if (cond_num_E < 0.0)  printf( "(no traction bc's -> no matrix needed)\n" );
	else {
		if (check_cond_num_E){
			if (cond_num_E < 1e5)  printf( "%.2f\n", cond_num_E );
			else                   printf( "%.4e\n", cond_num_E );
		}
		else  printf( "(not requested)\n" );
	}
	
	/* Print different time units depending on how long the code ran
	   Added by Scott T. Marshall 2015-10-18
	-------------------------------------------------------------------*/
	if      (elapsed_time < 60)    printf( "Total time    : %.0f seconds\n", elapsed_time);
	else if (elapsed_time < 3600)  printf( "Total time    : %.2f minutes\n", elapsed_time/60);
	else if (elapsed_time < 86400) printf( "Total time    : %.2f hours\n", elapsed_time/3600);
	else                           printf( "Total time    : %.2f days (%.2f hours)\n", elapsed_time/86400, elapsed_time/3600);
	printf( "-------------------------------------------------------------\n\n\n" );
	return(0);
}


/*********************** Function: print_problem_info ***********************
* Prints general problem information to the output file.
****************************************************************************/
void     print_problem_info(void)
{
	/* Print program name, version, and date
	----------------------------------------*/
	fprintf(ofp_E,"OUTPUT FROM: %s, version %s\n",PROGRAM, VERSION);
	fprintf(ofp_E,"   COMPILED: %s\n",COMPILE_DATE);

	/* Print input file name and problem titles
	-------------------------------------------*/
	fprintf(ofp_E,"\n INPUT FILE: %s\n",infile_E);
	fprintf(ofp_E,  "     TITLE1: %s\n",title1_E);
	fprintf(ofp_E,  "     TITLE2: %s\n",title2_E);

	/* Print elastic constant values
	--------------------------------*/
	fprintf(ofp_E,"\nELASTIC CONSTANTS:\n");
	fprintf(ofp_E,  "    Shear Modulus   = %f\n",shear_mod_E);
	fprintf(ofp_E,  "    Poisson's Ratio = %f\n",psn_ratio_E);
	fprintf(ofp_E,  "    Young's Modulus = %f\n",youngs_mod_E);
	fprintf(ofp_E,  "    Bulk Modulus    = %f\n",bulk_mod_E);
	fprintf(ofp_E,  "    Lame's Lambda   = %f\n",lame_lambda_E);

	/* Print the null output value
	------------------------------*/
	fprintf(ofp_E,"\nNULL OUPUT VALUE = %f\n",null_value_E);

    /* Print the coef exclusion value
	------------------------------*/
	fprintf(ofp_E,"\nCOEF EXCLUSION VALUE = %f\n",coef_exclu_E);

	/* Print condition number of the influence coefficient matrix
	-------------------------------------------------------------*/
	fprintf(ofp_E,"\nCONDITION NUMBER = ");
	if (cond_num_E < 0.0) {
		fprintf(ofp_E,"(no traction bc's -> no matrix needed)\n");
	} else {
		if (check_cond_num_E)
			fprintf(ofp_E,"%.2f\n",cond_num_E);
		else
			fprintf(ofp_E,"(not requested)\n");
	}
	
	/* Print element geometries (if requested)
	------------------------------------------*/
	if (print_elt_geom_E)
		print_elt_geometry();
}


/************************** Function: print_obj_data ************************
* Calculates and prints object data to the output file.
* PARALLELIZED VERSION
*****************************************************************************/
void     print_obj_data(void)
{
    obj_t   *current_obj;
    int     calc_displ;
    int     calc_strain;

    /* Loop through objects one by one */
    current_obj = first_obj_E;
    while (current_obj != NULL) {
        
        /* Check if we need to process this object */
        calc_displ  = current_obj->print[DISPL];
        calc_strain = current_obj->print[STRESS];

        if (calc_displ || calc_strain) {
            
            /* 1. SETUP: Open files and print headers (Serial) */
            open_temp_files(current_obj->print);
            print_obj_titles(current_obj);

            /* 2. GATHER: Count elements and build an array for parallel access */
            int num_elts = 0;
            elt_t *e = current_obj->first_elt;
            while (e != NULL) {
                num_elts++;
                if (e == current_obj->last_elt) break;
                e = e->next;
            }

            /* Allocate temporary buffers to hold results */
            elt_t **elt_ptr_array = (elt_t **) malloc(num_elts * sizeof(elt_t*));
            double (*displ_buf)[3] = (double (*)[3]) malloc(num_elts * 3 * sizeof(double));
            double (*stress_buf)[3][3] = (double (*)[3][3]) malloc(num_elts * 9 * sizeof(double));
            int *near_vertex_flags = (int *) malloc(num_elts * sizeof(int));

            /* Fill the pointer array */
            e = current_obj->first_elt;
            for (int i = 0; i < num_elts; i++) {
                elt_ptr_array[i] = e;
                e = e->next;
            }

            /* 3. COMPUTE: Calculate Math in Parallel */
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < num_elts; i++) {
                double local_displ[3];
                double local_strain[3][3];
                
                /* Perform the heavy math */
                displ_strain(calc_displ, calc_strain,
                    elt_ptr_array[i]->elt_csys.origin,
                    local_displ, local_strain,
                    elt_ptr_array[i]);
                
                /* Capture the thread-local global flag */
                near_vertex_flags[i] = near_vertex_E;

                /* Convert strain to stress */
                strain_to_stress(local_strain, shear_mod_E, lame_lambda_E, stress_buf[i]);

                /* Store displacement */
                memcpy(displ_buf[i], local_displ, 3 * sizeof(double));
            }

            /* 4. WRITE: Write to files Sequentially (Serial) */
            for (int i = 0; i < num_elts; i++) {
                /* Restore the global flag for the printing function */
                near_vertex_E = near_vertex_flags[i];

                /* Print using the original function */
                print_elt_data(current_obj, elt_ptr_array[i], i + 1,
                    displ_buf[i], stress_buf[i]);
            }

            /* 5. CLEANUP: Free buffers and close files */
            free(elt_ptr_array);
            free(displ_buf);
            free(stress_buf);
            free(near_vertex_flags);

            copy_temp_files();
            close_temp_files();
        }

        /* Move to next object */
        current_obj = current_obj->next;
    }
}


/********************** Function: print_obs_grid_data ******************
* Loops through linked list of observation grids, caculating and printing
* the requested displacement, strain, and stress data for each to the
* output file.
* PARALLELIZED VERSION
************************************************************************/
void     print_obs_grid_data(void)
{
    obs_grid_t  *current_obs_grid;
    int         i;
    double      dx[3];
    double      *begin;
    double      *end;
    int         *numpts;
    char        error_msg[MAX_ERROR_MSG];
    int         *print;
    int         calc_displ;
    int         calc_strain;

    /* Loop over each observation grid */
    current_obs_grid = first_obs_grid_E;
    while (current_obs_grid != NULL) {

        /* -------------------------------------------------------------------
           CASE A: 0-D Observation Grid (Single Point)
           Check for a batch of 0-D grids to process in parallel
           ------------------------------------------------------------------- */
        if (current_obs_grid->dimension == 0) {
            
            /* 1. LOOKAHEAD: Count consecutive 0-D grids */
            int batch_count = 0;
            obs_grid_t *scanner = current_obs_grid;
            while (scanner != NULL && scanner->dimension == 0) {
                batch_count++;
                scanner = scanner->next;
            }

            /* 2. ALLOCATE: Buffers for the batch */
            obs_grid_t **grid_batch = (obs_grid_t **) malloc(batch_count * sizeof(obs_grid_t*));
            double (*displ_buf)[3] = (double (*)[3]) malloc(batch_count * 3 * sizeof(double));
            double (*strain_buf)[3][3] = (double (*)[3][3]) malloc(batch_count * 9 * sizeof(double));
            double (*x_buf)[3] = (double (*)[3]) malloc(batch_count * 3 * sizeof(double));
            int *near_vertex_buf = (int *) malloc(batch_count * sizeof(int));

            if (!grid_batch || !displ_buf || !strain_buf || !x_buf || !near_vertex_buf) {
                p_error("Memory allocation failed for 0D grid batch processing", NULL);
            }

            /* Fill grid pointer array */
            scanner = current_obs_grid;
            for(i = 0; i < batch_count; i++) {
                grid_batch[i] = scanner;
                scanner = scanner->next;
            }

            /* 3. COMPUTE: Parallel Physics Calculation */
            #pragma omp parallel for schedule(dynamic)
            for(i = 0; i < batch_count; i++) {
                obs_grid_t *grid = grid_batch[i];
                double x[3];
                int calc_d, calc_s;

                /* Determine flags */
                calc_d = grid->print[DISPL];
                calc_s = (grid->print[STRAIN] || grid->print[PSTRAIN] || grid->print[STRESS] || grid->print[PSTRESS]);

                /* Calculate position (Transform logic) */
                copy_vector(grid->begin, x);
                transform_position_vector(INVERSE_TRANS, 
                    grid->endpt_csys->origin,
                    grid->endpt_csys->local_rot, x);
                
                /* Store global coord for printing */
                copy_vector(x, x_buf[i]);

                /* Compute Physics */
                displ_strain(calc_d, calc_s, x, displ_buf[i], strain_buf[i], NULL);
                
                /* Capture thread-local global flag */
                near_vertex_buf[i] = near_vertex_E;
            }

            /* 4. OUTPUT: Serial File Writing */
            for(i = 0; i < batch_count; i++) {
                obs_grid_t *grid = grid_batch[i];

                open_temp_files(grid->print);
                print_obs_grid_titles(grid);

                /* Restore the near_vertex_E flag for this specific point */
                near_vertex_E = near_vertex_buf[i];

                print_obs_pt_data(grid, x_buf[i], displ_buf[i], strain_buf[i]);

                copy_temp_files();
                close_temp_files();
            }

            /* 5. CLEANUP */
            free(grid_batch); 
            free(displ_buf); 
            free(strain_buf); 
            free(x_buf); 
            free(near_vertex_buf);

            /* Advance the main pointer past the batch */
            current_obs_grid = scanner; 

        } 
        /* -------------------------------------------------------------------
           CASE B: 1D, 2D, 3D Grids (Lines/Grids)
           Use existing logic which handles its own internal parallelization
           ------------------------------------------------------------------- */
        else {
            /* Determine data required by obs grid print options */
            print = current_obs_grid->print;
            calc_displ  = print[DISPL];
            calc_strain = (print[STRAIN] || print[PSTRAIN] || print[STRESS] || print[PSTRESS]);

            /* Remember that begin_pt & end_pt are in GLOBAL coordinate system.*/
            begin   = current_obs_grid->begin;
            end     = current_obs_grid->end;
            numpts  = current_obs_grid->numpts;
            subtract_vectors(end, begin, dx);

            /* Open temp files */
            open_temp_files(current_obs_grid->print);

            /* Print observation grid titles */
            print_obs_grid_titles(current_obs_grid);

            /* Process the observation grid */
            int total_pts = 0;
            double (*displ_buf)[3] = NULL;
            double (*strain_buf)[3][3] = NULL;
            double (*x_buf)[3] = NULL;
            int *near_vertex_buf = NULL;

            /* Determine total points */
            if (current_obs_grid->dimension == 1) total_pts = numpts[0];
            else total_pts = numpts[0] * numpts[1] * numpts[2];

            /* Allocate Memory */
            if (total_pts > 0) {
                displ_buf = (double (*)[3]) malloc(total_pts * 3 * sizeof(double));
                strain_buf = (double (*)[3][3]) malloc(total_pts * 9 * sizeof(double));
                x_buf = (double (*)[3]) malloc(total_pts * 3 * sizeof(double));
                near_vertex_buf = (int *) malloc(total_pts * sizeof(int));
                
                if (!displ_buf || !strain_buf || !x_buf || !near_vertex_buf) {
                    p_error("Memory allocation failed in print_obs_grid_data", NULL);
                }
            }

            switch (current_obs_grid->dimension) {
                /* Case 0 is handled above in the batch block */
                
                case 1: /* Line of points */
                    /* Pre-calculate steps */
                    dx[2] /= (numpts[0]-1);
                    dx[1] /= (numpts[0]-1);
                    dx[0] /= (numpts[0]-1);

                    /* COMPUTE PARALLEL */
                    #pragma omp parallel for schedule(static)
                    for (i = 0; i < numpts[0]; i++) {
                        double local_x[3];
                        
                        /* Recalculate position for this index */
                        local_x[2] = begin[2] + dx[2]*i;
                        local_x[1] = begin[1] + dx[1]*i;
                        local_x[0] = begin[0] + dx[0]*i;

                        /* Transform */
                        transform_position_vector(INVERSE_TRANS, 
                            current_obs_grid->endpt_csys->origin,
                            current_obs_grid->endpt_csys->local_rot, local_x);
                        
                        /* Save X for printing */
                        copy_vector(local_x, x_buf[i]);

                        /* Compute Physics */
                        displ_strain(calc_displ, calc_strain, local_x, 
                                     displ_buf[i], strain_buf[i], NULL);
                        
                        /* Capture thread-local global flag */
                        near_vertex_buf[i] = near_vertex_E;
                    }
                    
                    /* WRITE SERIAL */
                    for (i = 0; i < numpts[0]; i++) {
                        near_vertex_E = near_vertex_buf[i]; /* Restore flag */
                        print_obs_pt_data(current_obs_grid, x_buf[i], displ_buf[i], strain_buf[i]);
                    }
                    
                    free(displ_buf); free(strain_buf); free(x_buf); free(near_vertex_buf);
                    break;

                case 2:
                case 3: /* 2D or 3D Grid */
                    for (i = 0; i < 3; i++) {
                        dx[i] /= (numpts[i] == 1) ? 1 : (numpts[i]-1);
                    }

                    /* COMPUTE PARALLEL - Flattened Loop */
                    #pragma omp parallel for schedule(dynamic)
                    for (int p = 0; p < total_pts; p++) {
                        int idx_i, idx_j, idx_k, rem;
                        double local_x[3];

                        /* Decode 1D index 'p' back to 3D indices */
                        idx_i = p / (numpts[1] * numpts[0]);
                        rem   = p % (numpts[1] * numpts[0]);
                        idx_j = rem / numpts[0];
                        idx_k = rem % numpts[0];

                        /* Calculate Position */
                        local_x[2] = begin[2] + dx[2]*idx_i;
                        local_x[1] = begin[1] + dx[1]*idx_j;
                        local_x[0] = begin[0] + dx[0]*idx_k;

                        /* Transform */
                        transform_position_vector(INVERSE_TRANS, 
                            current_obs_grid->endpt_csys->origin,
                            current_obs_grid->endpt_csys->local_rot, local_x);

                        /* Save X for printing */
                        copy_vector(local_x, x_buf[i]); /* Note: originally x_buf[p] in previous code, checking context... */
                        /* The variable 'i' is the loop variable for outer loops, here 'p' is the loop variable. */
                        /* Correcting index to 'p' to match the loop */
                         copy_vector(local_x, x_buf[p]);


                        /* Compute Physics */
                        displ_strain(calc_displ, calc_strain, local_x, 
                                     displ_buf[p], strain_buf[p], NULL);
                        
                        /* Capture thread-local global flag */
                        near_vertex_buf[p] = near_vertex_E;
                    }

                    /* WRITE SERIAL */
                    for (int p = 0; p < total_pts; p++) {
                        near_vertex_E = near_vertex_buf[p]; /* Restore flag */
                        print_obs_pt_data(current_obs_grid, x_buf[p], displ_buf[p], strain_buf[p]);
                    }

                    free(displ_buf); free(strain_buf); free(x_buf); free(near_vertex_buf);
                    break;

                default:
                    sprintf(error_msg, "Invalid dimension (%d) for observation grid", current_obs_grid->dimension);
                    p_error(error_msg, NULL);
            }

            /* Copy temp files to main output file, then close them. */
            copy_temp_files();
            close_temp_files();

            current_obs_grid = current_obs_grid->next;
        }
    }
}


/************************ Function: displ_strain ****************************
* Calculates the total displacement and/or strain at a point due to ALL
* elements. 
*
* In:	   calc_displ	- calculate displacements flag
*		   calc_strain - calculate strains flag
*		   x			   - coords (global) of pt at which to calc displ & strain
*		   omit_elt	   - element to omit when calculating displs (NULL = none)
*
* Out:	displ  		- displacement vector (global coords)
*		   strain 		- strain tensor (global coords)
*
*****************************************************************************/
void     displ_strain(int calc_displ, int calc_strain, double x[3],double displ[3], double strain[3][3], elt_t *omit_elt)
{
	elt_t	*current_elt;
	double	elt_displ[3];
	double	elt_strain[3][3];
/*Declarations for correction of the "shadow effect" */
         int i;
         double orient;
      	double under_plane;
      	int under;
      	double normal [3];
      	double data1 [3];
      	double data [3];
      	vert_t *verta;
      	vert_t *vertb;
      	vert_t *vertc;
      	double seg1 [3];
      	double seg2 [3];
      	double inside;
      	double inside_vector [3];
      	int inside_test=0;
      	int inside_test_fin;
      	double x3_global [3];
      	disloc_seg_t *disloc_seg;
/* end of the declaration */

/*Declarations for coef_exclu */
         double dist;
/* end of the declaration */

	initialize_vector(displ,0.0);
	initialize_matrix(strain,0.0);

	/* Loop over each element
	-------------------------*/
	current_elt = first_elt_E;
   near_vertex_E = FALSE;
	while (current_elt != NULL)
   {
		/* Test UNDER to determine if the data point is inside the "shadow zone"
		---------------------------------------------------------------------*/
		under = 0;
		inside_test_fin = 1;
		
		/* Determine if the element has a positive side up or down (orient)
		   and determine if the data is under the plane 
			defined by the element (under_plane) 
		-----------------------------------------------------------------*/
		disloc_seg = current_elt->disloc_seg;
		x3_global [0] = 0;
		x3_global [1] = 0;
		x3_global [2] = 1;
		verta = disloc_seg [0].vert[0];
		vertb = disloc_seg [0].vert[1];
		vertc = disloc_seg [1].vert[1]; 
		subtract_vectors (vertb->x, verta->x, seg1);
		subtract_vectors (vertc->x, verta->x, seg2);
		cross_product (seg1, seg2, normal);
		orient = dot_product (normal, x3_global);
		subtract_vectors (x, verta->x, data1);
		under_plane = dot_product (normal, data1);

      /* Determine if obs_pt is nearnest vertex verta, vertb or vertc
         1) compute d = mean length of the 3 disloc_seg
         2) compute distance from obs_pt to verta and see if this diance is < d*coef_exclu
         3) compute distance from obs_pt to vertb and see if this diance is < d*coef_exclu
         4) compute distance from obs_pt to vertc and see if this diance is < d*coef_exclu
         if condition is TRUE, break the loop over elements and set the flag near_vertex_E=TRUE
            and return.
            
         We use a new function "distance" define in matrix.h & matrix.c
      */
      dist = 0.0;
      dist  = distance(verta->x,vertb->x);
      dist += distance(vertb->x,vertc->x);
      dist += distance(vertc->x,verta->x);
      dist *= coef_exclu_E/3.0;
      
      if (distance(x,verta->x)<=dist || distance(x,vertb->x)<=dist || distance(x,vertc->x)<=dist)
      {
         near_vertex_E = TRUE;
         break;
      }
		
		/* Determine if the data point is in the "rigid body" 
			(cf. explanation of the bug)
		---------------------------------------------------*/

		for (i = 0; i < current_elt->num_vertices; i++)
		{
			inside = 0;
			verta = disloc_seg [i].vert[0];
			vertb = disloc_seg [i].vert[1];
			subtract_vectors (vertb->x, verta->x, seg1);
		   subtract_vectors (x, verta->x, data);
			cross_product (seg1, data, inside_vector);
			inside = dot_product (x3_global, inside_vector);
			
			if (orient > 0) 
			{
               if (inside > 0 && under_plane < 0)
                  inside_test = 1;
				   else
                  inside_test = 0;
			}
			if (orient < 0)
			{
            if (inside < 0 && under_plane >  0)
               inside_test = 1;
				else
               inside_test = 0;
			}
			if (orient == 0)
				under = 0;

			inside_test_fin *= inside_test;  
		}

		/*Gives a value 1 to under for data points under the element
		and positive side up
		Gives a value 2  to under for data points under the element
		and positive side down
		Gives a value 0  to under for data points not under the element
		---------------------------------------------------------------*/

		if (inside_test_fin > 0)
      {
			if (orient > 0)
            under = 1;
		   if (orient < 0)
            under = 2;
		}
      else
         under = 0;

		/* Calculate displacement and strain due to this element
		--------------------------------------------------------*/
		displ_strain_poly_elt(calc_displ,calc_strain,current_elt,
			x,elt_displ,elt_strain,omit_elt,under);

		/* Add the contribution of this element to the total displ & strain
		-----------------------------------------------------------------*/
		add_vectors(displ,elt_displ,displ);
		add_matrices(strain,elt_strain,strain);
		current_elt = current_elt->next;
	}

	/* Adjust strain for remote strain
	----------------------------------*/
   if (near_vertex_E!=TRUE)
	   add_matrices(strain,rem_strain_E,strain);
}


/****************** Function: displ_strain_poly_elt ********************
* Calculates the displacement and/or strain at a point due to a single
* polygonal element.
*
* In:	calc_displ	- calculate displacements flag
*		calc_strain - calculate strains flag
*		current_elt	- element being considered
*		x			- coords (global) of pt at which to calc displ & strain
*		omit_elt	- element to omit when calculating displs (NULL = none)
* Out:	elt_displ  	- displacement vector (global coords) due to this elt
*		elt_strain	- strain tensor (global coords) due to this elt
****************************************************************************/
void     displ_strain_poly_elt(int calc_displ, int calc_strain,elt_t *current_elt, double x[3], double elt_displ[3],double elt_strain[3][3], elt_t *omit_elt, int under)
{
	double	displ_ic[3][3];
	double	strain_ic[3][3][3];
	int		i, j, k;
	/*Declarations for shadow effect correction*/
	double bglobal[3];
	double e[3][3];
	double eg[3][3];
	double bg[3][3];

	/* Initialize displacement vector and strain tensor
	---------------------------------------------------*/
	initialize_vector(elt_displ,0.0);
	initialize_matrix(elt_strain,0.0);

	/* Calculate the displacement and strain influence coefficients
	---------------------------------------------------------------*/
	displ_strain_ics_poly_elt(calc_displ,calc_strain,current_elt,x,
		displ_ic,strain_ic,omit_elt);

	/* Superpose the contribution from each burger's vector component
	-----------------------------------------------------------------*/
	for (i=0; i < 3; i++)
   {
		for (j=0; j < 3; j++)
      {
			elt_displ[i] += displ_ic[j][i] * (*current_elt->b[j]); 	
         for (k=0; k < 3; k++)
				elt_strain[i][j] += strain_ic[k][i][j] * (*current_elt->b[k]);
		}
	}

 /* Shadow effect correction in case of data point under the element
---------------------------------------------------------------------------------*/
		
	if (under > 0)
   {
	   bglobal[0]=0;
      bglobal[1]=0;
      bglobal[2]=0;

		e[0][0]=1;e[0][1]=0;e[0][2]=0;
		e[1][0]=0;e[1][1]=1;e[1][2]=0;
		e[2][0]=0;e[2][1]=0;e[2][2]=1;

		eg[0][0]=1;eg[0][1]=0;eg[0][2]=0;
		eg[1][0]=0;eg[1][1]=1;eg[1][2]=0;
		eg[2][0]=0;eg[2][1]=0;eg[2][2]=1;

		for (i=0; i < 3; i++)
      {
			/* Transform the burger vector's component in the global coordinate system
			--------------------------------------------------------------------------- */
			rotate_vector(INVERSE_ROT,current_elt->elt_csys.local_rot,e[i]);
			scalar_vector_mult (*current_elt->b[i], e[i],bg[i]);
		}
		for (i=0; i < 3; i++)
      {
		   for (j=0; j<3; j++)
            bglobal[i]+=dot_product(bg[j], eg[i]);

			/*Corrects the displacement by the corresponding burger's vector (global coordsys)
			--------------------------------------------------------------------------------*/ 
			if (under == 1)
			   elt_displ[i] -= bglobal[i];
         else
			if (under == 2)
				elt_displ[i] += bglobal[i];
		}

	}/* end of if (under>0)*/
}

/***************** Function: displ_strain_ics_poly_elt **********************
* Calculates the displacement and/or strain influence coefficients at a point 
* due to a polygonal element.
*
* In:	calc_displ	- calculate displacements flag
*		calc_strain - calculate strains flag
*		current_elt	- element being considered
*		x			- coords (global) of pt at which to calc displ & strain
*		omit_elt	- element to omit when calculating displs (NULL = none)
*
* Out:	displ_ic[i][j]		- the jth component of displ (global coords)
*                        	  due to a unit ith Burgers vector component 
*                        	  (bc_coord_sys coords)
*		strain_ic[i][j][k]	- the jk component of strain (global coords)
*                        	  due to a unit ith Burgers vector component 
*                        	  (bc_coord_sys coords)
****************************************************************************/
void     displ_strain_ics_poly_elt(int calc_displ, int calc_strain,elt_t *current_elt, double x[3], double displ_ic[3][3],double strain_ic[3][3][3], elt_t *omit_elt)
{
	int		i, j, k, l;
	int		seg;
	int		swap;
	vert_t	 *vert1;
	vert_t	 *vert2;
	double	depth1;
	double	depth2;
	double	beta;
	double	temp_double;
	double	temp_vector[3];
	double	r;
	double	z3;
	double	y1[3];
	double	y2[3];
	double	displ_ic1[3][3];
	double	displ_ic2[3][3];
	double	displ_ic3[3][3];
	double	displ_ic4[3][3];
	double	strain_ic1[3][3][3];
	double	strain_ic2[3][3][3];
	double	strain_ic3[3][3][3];
	double	strain_ic4[3][3][3];
	disloc_seg_t	 *disloc_seg;
	
	/* Initialize the influence coefficients
	----------------------------------------*/
	for (i=0; i < 3; i++) {
		initialize_vector(displ_ic[i],0.0);
		initialize_matrix(strain_ic[i],0.0);
	}

	/* Loop over the element's dislocation segments
	-----------------------------------------------*/
	disloc_seg = current_elt->disloc_seg;
	for (seg = 0; seg < current_elt->num_vertices; seg++) {

		/* Determine the segment vertices
		---------------------------------*/
		vert1 = disloc_seg[seg].vert[0];
		vert2 = disloc_seg[seg].vert[1];

		/* Compute the vectors from the segments vertices to the
		  	data point
		---------------------------------------------------------*/
		subtract_vectors(x,vert1->x,y1);
		subtract_vectors(x,vert2->x,y2);

		/* Rotate vert-to-obs_point vectors to segment-local coords
		-----------------------------------------------------------*/
		rotate_vector(FORWARD_ROT,disloc_seg[seg].local_rot,y1);
		rotate_vector(FORWARD_ROT,disloc_seg[seg].local_rot,y2);

		depth1 = -vert1->x[2];
		depth2 = -vert2->x[2];
		beta   = PI/2.0 - disloc_seg[seg].plunge;

		if (((sqrt(y1[0]*y1[0]+y1[1]*y1[1]) < BVERT_TINY) && y1[2] >= 0.0) ||
			((sqrt(y2[0]*y2[0]+y2[1]*y2[1]) < BVERT_TINY) && y2[2] >= 0.0)) {
			below_vertex_E = TRUE;
			return;
		}

		/* If x lies along dipping leg of angular dislocations, swap
		   the vertex order, so singularity will be avoided
		------------------------------------------------------------*/
		swap = FALSE;
		z3 = y1[0]*sin(beta) + y1[2]*cos(beta);
		r  = vector_magnitude(y1);
		if ((r - z3) < SWAP_TINY) {
			swap = TRUE;
			copy_vector(y2,temp_vector);
			copy_vector(y1,y2);
			copy_vector(temp_vector,y1);
			for (i=0; i < 2; i++) {
				y1[i] *= -1.0;
				y2[i] *= -1.0;
			}
			temp_double = depth2;
			depth2 = depth1;
			depth1 = temp_double;
			beta = PI - beta;
		} 
 
		/* Be carefull for the calculation of vertical elements 
		------------------------------------------------------*/ 
		if (beta==0.0) 
			beta = 1.0e-14;
			

		/* Calculate displacement influence coeffs
		------------------------------------------*/
		if (calc_displ) {

			/* Avoid displacement discontinuity when calculating displ
			   inf coeff of an element on itself
			----------------------------------------------------------*/
			if  (current_elt != omit_elt) {

				comninou_displ_ics(y1,depth1,beta,psn_ratio_E,
					half_space_E,displ_ic1);
				comninou_displ_ics(y2,depth2,beta,psn_ratio_E,
					half_space_E,displ_ic2);

				/* Superpose the angular dislocation influence coeffs into
		   	   	a dislocation segment influence coeff
				----------------------------------------------------------*/
				subtract_matrices(displ_ic1,displ_ic2,
					displ_ic3);

				/* Swap the vertices back to proper order (if necessary)
				--------------------------------------------------------*/
				if (swap) {
					scalar_vector_mult(-1.0,displ_ic3[2],
						displ_ic3[2]);
					for (i=0; i < 3; i++) {
						displ_ic3[i][0] *= -1.0;
						displ_ic3[i][1] *= -1.0;
					}
				}

				/* Transform from disloc segment to element influence coeffs
				------------------------------------------------------------*/
				for (i=0; i < 3; i++) {
					initialize_vector(displ_ic4[i],0.0);
					for (j=0; j < 3; j++) {
						for (k=0; k < 3; k++) {
							displ_ic4[i][j] +=
								disloc_seg[seg].elt_b[i][k] *
								displ_ic3[k][j];
						}
					}
				}
		
				/* Rotate from C&D to global coordinates
				----------------------------------------*/
				for (i=0; i < 3; i++) {
					rotate_vector(INVERSE_ROT,disloc_seg[seg].local_rot,
						displ_ic4[i]);
				}

				/* Superpose the contribution of this dislocation segment
				---------------------------------------------------------*/
				for (i=0; i < 3; i++) {
					add_vectors(displ_ic[i],displ_ic4[i],
						displ_ic[i]);
				}
	
			} 
		}

		/* Calculate strain influence coeffs
		------------------------------------*/
		if (calc_strain) {

			comninou_strain_ics(y1,depth1,beta,psn_ratio_E,
				half_space_E,strain_ic1);
			comninou_strain_ics(y2,depth2,beta,psn_ratio_E,
				half_space_E,strain_ic2);

			/* Superpose the angular dislocation influence coeffs into
		   	   a dislocation segment influence coeff
			----------------------------------------------------------*/
			for (i=0; i < 3; i++) {
				subtract_matrices(strain_ic1[i],strain_ic2[i],
					strain_ic3[i]);
			}

			/* Swap the vertices back to proper order (if necessary)
			--------------------------------------------------------*/
			if (swap) {
				scalar_matrix_mult(-1.0,strain_ic3[2],
					strain_ic3[2]);
				for (i=0; i < 3; i++) {
					strain_ic3[i][0][2] *= -1;
					strain_ic3[i][2][0] *= -1;
					strain_ic3[i][1][2] *= -1;
					strain_ic3[i][2][1] *= -1;
				}
			}


			for (i=0; i < 3; i++) {
				initialize_matrix(strain_ic4[i],0.0);
				for (j=0; j < 3; j++) {
					for (k=0; k < 3; k++) {
						for (l=0; l < 3; l++) {
							strain_ic4[i][j][k] +=
								disloc_seg[seg].elt_b[i][l] *
								strain_ic3[l][j][k];
						}
					}
				}
			}
		
			/* Rotate strain inf coeffs to global coords
			--------------------------------------------*/
			for (i=0; i < 3; i++) {
				rotate_tensor(INVERSE_ROT,disloc_seg[seg].local_rot, strain_ic4[i]);
			}

			/* Superpose the contribution of this dislocation segment
			---------------------------------------------------------*/
			for (i=0; i < 3; i++) {
				add_matrices(strain_ic[i],strain_ic4[i],
					strain_ic[i]);
			}

		}
	
	} /* loop over dislocation segments */

}


/**************************** Function: read_infile **************************
* Reads the input file and sets up the problem to be solved.
******************************************************************************/
void     read_infile(void)
{

	/* Read problem constants
	-------------------------*/
	read_constants();

	/* Read coordinate systems
	--------------------------*/
	read_csystems();

	/* Read observation grids
	-------------------------*/
	read_observation_grids();

	/* Read elements and vertices
	-----------------------------*/
	read_objs_elts_verts();

}


/************************** Function: find_csys *************************
* Returns a pointer to the coordinate system named by name, or NULL if no
* such coordinate system exists.
*
* In:	name	- name of the coordinate system to find
*****************************************************************************/
csys_t*  find_csys(const char *name)
{
	csys_t *current_csys;

	current_csys = first_csys_E;

	while (current_csys != NULL) {
		if (!strcmp(name,current_csys->name))
			break;
		current_csys = current_csys->next;
	}

	return(current_csys);
}


/***************************** Function: find_vert ***************************
* Return a pointer to the vertex named by name, or NULL if no such vertex
* exists.
*
* In:	name	- name of the vertex to find
*****************************************************************************/
vert_t*  find_vert(const char *name)
{
	vert_t	*current_vert;

	current_vert = first_vert_E;

	while (current_vert != NULL) {
		if (!strcmp(name,current_vert->name))
			break;
		current_vert = current_vert->next;
	}

	return(current_vert);
}


/***************************** Function: p_error *****************************
* Prints an error message to stderr and calls exit().  If line != NULL, the
* line and line number (from the input file) on which the error occured are
* printed as well.
*
* In:	error_msg	- error message to print
* 		line		- input file line number at which the error occurred
*					  (NULL = N/A)
*****************************************************************************/
void     p_error(const char *error_msg, const char *line)
{
	fprintf(stderr,"\nerror: %s",error_msg);
	if (line == NULL) {
		fprintf(stderr,"\n");
	} else {
		fprintf(stderr," (%s, line %d)\n",infile_E,linenum_E);
		fprintf(stderr,"       %s\n",line);
	}
	exit(1);

}

/***************************** Function: display_msg *****************************
* Prints an message to stderr.
*
* In:	error_msg	- error message to print
*****************************************************************************/
void     display_msg(char *_msg)
{
	fprintf(stderr,"%s\n",_msg);
}


/************************ Function: setup_global_coords **********************
* Defines the global coordinate system, making it the first member in the
* linked list of coordinate systems (first_csys_E).
*****************************************************************************/
void     setup_global_coords(void)
{

	int		i;

	/* Set first coord system to global coordinates
	-----------------------------------------------*/
	first_csys_E = (csys_t *) calloc((size_t) 1,
		sizeof(csys_t));
	if (!first_csys_E)
		p_error("Cannot allocate memory (calloc) for global coord sys",
			NULL);

	first_csys_E->name = (char *) calloc((size_t) 1,
		strlen(GLOBAL_NAME)+1);
	if (!first_csys_E->name)
		p_error( "Cannot allocate memory for global coord system name",
			NULL);
	strcpy(first_csys_E->name,GLOBAL_NAME);

	for (i=0; i < 3; i++) {
		first_csys_E->origin[i] = 0;
	}
	initialize_matrix(first_csys_E->local_rot,0.0);
	for (i=0; i < 3; i++) {
		first_csys_E->local_rot[i][i] = 1.0;
	}

}


/*************************** Function: open_files ****************************
* Opens the input and output files named by the external variables infile_E
* and outfile_E
*****************************************************************************/
int      open_files(void)
{
	char error_msg[MAX_ERROR_MSG];

	if (infile_E[0] != '\0') {
		if ((ifp_E = fopen(infile_E,"r")) == NULL) {
			snprintf(error_msg, sizeof(error_msg), "Cannot open the input file %s", infile_E);
			p_error(error_msg,NULL);
		} /*if*/
	} /*if*/

	if (outfile_E[0] != '\0') {
		if ((ofp_E = fopen(outfile_E,"w")) == NULL) {
			snprintf(error_msg, sizeof(error_msg), "Cannot open the output file %s", outfile_E);
			p_error(error_msg,NULL);
		} /*if*/
	} /*if*/

	return(0);
}


/********************** Function: parse_command_line_args ********************
* Parses the command line argument using getoptPoly3D().
*
* In:	argc	- number of command line arguments
*		argv	- the command line arguments
*****************************************************************************/
int      parse_command_line_args(int argc, char *argv[])
{
	int		optarg;						/* command line option/argument		*/
	int		exit = FALSE;
	
	/* CASE 1: No arguments provided */
    if (argc == 1) {
        fprintf(stderr,"\n");
		fprintf(stderr,"------------------------------------------------------\n");
		fprintf(stderr,"%s version %s\n",PROGRAM,VERSION);
		fprintf(stderr,"------------------------------------------------------\n");
		fprintf(stderr,"Original v0.0 : Andrew L. Thomas  - Jun, 1993\n");
		fprintf(stderr,"Parallel v3.0 : Scott T. Marshall - Jan, 2026\n");
		fprintf(stderr,"------------------------------------------------------\n");
		fprintf(stderr,"Compiled  : %s by Scott T. Marshall\n",COMPILE_DATE);
		fprintf(stderr,"Questions : marshallst@appstate.edu\n");
		fprintf(stderr,"------------------------------------------------------\n");
		fprintf(stderr,"\nUsage: %s -i infile -o outfile",PROGRAM);
		fprintf(stderr,"\n  (arguments may occur in any order)\n\n\n");
        return(ERROR); /* Returns ERROR so main() exits */
    }
	
	infile_E[0] = outfile_E[0] = '\0';
	while ((optarg = getoptPoly3D("i:o:",argc,argv)) != NO_MORE_ARGS && !exit) {
		switch (optarg) {
			case NO_SUCH_ARG:
			case FILE_ARG:
				exit = TRUE;
				break;
			/* -i <filename> names the input file */
			case 'i':
				if (strlen(getopt_arg_E) > MAXFILE-1) {
					p_error("Input file name too long",NULL);
				}
				strcpy(infile_E,getopt_arg_E);
				break;
			/* -o <filename> names the output file */
			case 'o':
				if (strlen(getopt_arg_E) > MAXFILE-1) {
					p_error("Output file name too long",NULL);
				}
				strcpy(outfile_E,getopt_arg_E);
				break;
		} /*switch*/
	} /*while*/

	if (exit) {
		/*original message is below. Added new info 2015-10, updated again 2025-12 */
		/*fprintf(stderr,"\nUsage: poly3d [-i infile] [-o outfile]"); */
		fprintf(stderr,"\n");
		fprintf(stderr,"------------------------------------------------------\n");
		fprintf(stderr,"%s version %s\n",PROGRAM,VERSION);
		fprintf(stderr,"------------------------------------------------------\n");
		fprintf(stderr,"Original v0.0 : Andrew L. Thomas  - Jun, 1993\n");
		fprintf(stderr,"Parallel v3.0 : Scott T. Marshall - Jan, 2026\n");
		fprintf(stderr,"------------------------------------------------------\n");
		fprintf(stderr,"Compiled  : %s by Scott T. Marshall\n",COMPILE_DATE);
		fprintf(stderr,"Questions : marshallst@appstate.edu\n");
		fprintf(stderr,"------------------------------------------------------\n");
		fprintf(stderr,"\nUsage: %s -i infile -o outfile",PROGRAM);
		fprintf(stderr,"\n  (arguments may occur in any order)\n\n\n");
        return(ERROR); /* Returns ERROR so main() exits */
				
	} /*if*/

	return(0);
}


/*************************** Function: read_constants ************************
* Reads problem constants from the input file, skipping blank and comment
* lines.  Stops reading when a line beginning with END_STMT is reached.
* Sets up remote stress and strain tensors and calls calc_elas_consts() to
* calculate undefined elastic constants.
*****************************************************************************/
int      read_constants(void)
{
	int		numwords;
	char	*word[MAXWORDS];
	char	line[MAXLINE];
	char	error_msg[MAX_ERROR_MSG];
	int		temp;
	double	s11r, s22r, s33r, s12r, s13r, s23r;

	s11r = s22r = s33r = s12r = s13r = s23r = 0.0;

	/* read list of constants
	-------------------------*/ 
	for (;;) {

		/* read line from input file, increment linenum
		-----------------------------------------------*/
		numwords = read_line(line,word);
		linenum_E++;

		/* skip blank and comment lines
		-------------------------------*/
		if (numwords == 0) 
			continue;

		/* exit loop when end of list reached
		-------------------------------------*/ 
		if (!strcmp(word[0],END_STMT))
			break;

		/* parse constants
		------------------*/

		if (get_text_var(&title1_E,"title1",(const char * const *)word,numwords,line))
			continue;

		if (get_text_var(&title2_E,"title2",(const char * const *)word,numwords,line))
			continue;

		if (get_double_var(&shear_mod_E,"shear_mod",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&psn_ratio_E,"psn_ratio",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&youngs_mod_E,"youngs_mod",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&bulk_mod_E,"bulk_mod",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&lame_lambda_E,"lame_lambda",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&null_value_E,"null_value",(const char * const *)word,numwords))
			continue;
/*********************************************** ADDED 98-12-09 */
      if (get_double_var(&coef_exclu_E,"coef_exclu",(const char * const *)word,numwords))
			continue;
/***************************************************************/

		if (get_boolean_var(&rem_stress_bc_E,"rem_bc_type","stress","strain",(const char * const *)word,numwords,line))
			continue;

		if (get_double_var(&s11r,"s11r",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&s22r,"s22r",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&s33r,"s33r",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&s12r,"s12r",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&s13r,"s13r",(const char * const *)word,numwords))
			continue;

		if (get_double_var(&s23r,"s23r",(const char * const *)word,numwords))
			continue;

		if (get_boolean_var(&half_space_E,"half_space","yes","no",(const char * const *)word,numwords,line))
			continue;

		if (get_boolean_var(&check_cond_num_E,"check_cond_num","yes","no",(const char * const *)word,numwords,line))
			continue;

		if (get_boolean_var(&print_elt_geom_E,"print_elt_geom","yes","no",(const char * const *)word,numwords,line))
			continue;

		if (get_text_var(&elt_geom_csys_name_E,"elt_geom_csys",(const char * const *)word,numwords,line))
			continue;

		/* Print error message if line has incorrect format
		---------------------------------------------------*/
		else {
			p_error("Unknown constant, or incorrect format",line);
		}
	}

	/* Calculate elastic constants
	------------------------------*/
	if ((temp = calc_elas_consts(&shear_mod_E, &psn_ratio_E, &youngs_mod_E,
		&bulk_mod_E, &lame_lambda_E)) != 0) {
		sprintf(error_msg,"Too %s elastic constants defined (define two)",
			(temp == EC_TOO_FEW) ? "few" : "many");
		p_error(error_msg,NULL);
	} 

	/* Set up remote boundary condition stress and strain tensors
	-------------------------------------------------------------*/
	rem_stress_E[0][0] = s11r;
	rem_stress_E[1][1] = s22r;
	rem_stress_E[2][2] = s33r;
	rem_stress_E[0][1] = rem_stress_E[1][0] = s12r;
	rem_stress_E[0][2] = rem_stress_E[2][0] = s13r;
	rem_stress_E[1][2] = rem_stress_E[2][1] = s23r;
	copy_matrix(rem_stress_E,rem_strain_E);
	if (rem_stress_bc_E) {
		stress_to_strain(rem_strain_E,youngs_mod_E,psn_ratio_E,
			rem_strain_E);
	}
	else {
		strain_to_stress(rem_stress_E,shear_mod_E,lame_lambda_E,
			rem_stress_E);
	}

	return(0);
}


/************************ Function: read_csystems ***********************
* Reads user coordinate systems from the input file, skipping blank and
* comment lines.  Stops reading when a line beginning with END_STMT is
* reached.  Adds coordinate systems to the linked list whose first member
* (global coords) is given by first_csys_E.
*****************************************************************************/
int      read_csystems(void)
{
	int		numwords;
	char	*word[MAXWORDS];
	char	error_msg[MAX_ERROR_MSG];
	char	line[MAXLINE];
	char	temp_char;
	int		i,j,k;
	double	rot_matrix[3][3][3];
	double	local_rot[3][3];
	int		rot_order[3];
	double	rot[3];
	csys_t	*parent;
	csys_t	*current_csys;
	

	/* Initialize coordinate rotation matrices
	------------------------------------------*/
	initialize_matrix(local_rot,0.0);
	for (i=0; i < 3; i++) {
		initialize_matrix(rot_matrix[i],0.0);
	}

	/* Set first coordinate system to global coordinates
	----------------------------------------------------*/
	setup_global_coords();
	current_csys = first_csys_E;
			
	/* Read in coordinate systems
	-----------------------------*/
	for (;;) {

		/* read line from input file, increment linenum
		-----------------------------------------------*/
		numwords = read_line(line,word);
		linenum_E++;

		/* Skip blank and comment lines
		-------------------------------*/
		if (numwords == 0)
			continue;

		/* Exit loop when end of list reached
		-------------------------------------*/ 
		if (!strcmp(word[0],END_STMT))
			break;

		/* Check for proper number of parameters
		----------------------------------------*/
		else if (numwords != CS_NUM_PARAMS) {
			sprintf(error_msg,
				"Too %s parameters specified to define coord system",
				(numwords < CS_NUM_PARAMS) ? "few" : "many");
			p_error(error_msg,line);
		}

		/* Check if coordinate system name already taken
		------------------------------------------------*/
		if (find_csys(word[CS_NAME_POS]) != NULL) {
			p_error("Coordinate system already exits",line);
		}

		/* Allocate memory in linked list for local coordinate system
		-------------------------------------------------------------*/
		current_csys->next = (csys_t *)
			calloc((size_t) 1,sizeof(csys_t));
		if (!current_csys->next)
			p_error("Cannot allocate memory (calloc) for local coord sys",
				line);
		current_csys = current_csys->next;

		/* Assign coordinate system name
		--------------------------------*/
		current_csys->name = (char *) malloc((size_t) 
			strlen(word[CS_NAME_POS])+1);
		if (!current_csys->name)
			p_error("Cannot allocate memory for coord system name",
			 line);
		strcpy(current_csys->name,word[CS_NAME_POS]);
		
		/* Get coordinate system parent
		-------------------------------*/
		if ((parent = 
			find_csys(word[CS_PARENT_POS])) == NULL) {
			p_error("Undefined coordinate system",line);
		}

		/* Get coordinate system origin and convert to global coords
		------------------------------------------------------------*/
		for (i=0; i < 3; i++) {
			current_csys->origin[i] = atof(word[CS_ORIGIN_POS+i]);
		}
		transform_position_vector(INVERSE_TRANS,
			parent->origin,parent->local_rot,current_csys->origin);

		/* Get rots about x1,x2,x3 axes of parent and convert to radians
		----------------------------------------------------------------*/
		for (i=0; i < 3; i++) {
			rot[i] = RADIANS(atof(word[CS_ROT_POS+i]));
		}

		/* Get the rotation order
		-------------------------*/
		for (i=0; i < 3; i++) {
			temp_char = word[CS_ROT_ORDER_POS][i];
			if (temp_char < '1' || temp_char > '3') {
				p_error("Invalid axis for rotation order",
					line);
			}
			rot_order[i] = temp_char - '1';
		}

		/* Set up the rotation matrices
		-------------------------------*/
		for (i=0; i < 3; i++) {
			j = i+1;
			k = i+2;
			if (j > 2) j -= 3;
			if (k > 2) k -= 3;
			rot_matrix[i][i][i] = 1.0;
			rot_matrix[i][j][j] =
				rot_matrix[i][k][k] = cos(rot[i]);
			rot_matrix[i][k][j] = 
				-(rot_matrix[i][j][k] =  sin(rot[i]));
		}

		/* Calculate the global-to-local coordinate rotation matrix
		-----------------------------------------------------------*/
		matrix_mult(rot_matrix[rot_order[0]],parent->local_rot,local_rot);
		matrix_mult(rot_matrix[rot_order[1]],local_rot,local_rot);
		matrix_mult(rot_matrix[rot_order[2]],local_rot,local_rot);
		copy_matrix(local_rot,current_csys->local_rot);

	}

	return(0);
}


/********************** Function: read_observation_grids ********************
* Reads observation grids from the input file, skipping blank and
* comment lines.  Stops reading when a line beginning with END_STMT is
* reached.  Adds observation grids to the linked list whose first member
* (global coords) is given by first_obs_grid_E.
*****************************************************************************/
int      read_observation_grids(void)
{
	int		numwords;
	char	*word[MAXWORDS];
	char	line[MAXLINE];
	int		i;
	int		dimension;
	char	error_msg[MAX_ERROR_MSG];
	int		correct_num_params;
	obs_grid_t	 *current_obs_grid=NULL;
	int		num_ones;
	int		numpts;
	char	temp_char;

	/* Read in observation grids
	----------------------------*/
	for (;;) {

		/* read line from input file, increment linenum
		-----------------------------------------------*/
		numwords = read_line(line,word);
		linenum_E++;

		/* skip blank and comment lines
		-------------------------------*/
		if (numwords == 0)
			continue;

		/* exit loop when end of list reached
		-------------------------------------*/ 
		if (!strcmp(word[0],END_STMT))
			break;

		/* Get grid dimension
		---------------------*/
		dimension = atoi(word[OG_DIMEN_POS]);

		/* Check for proper number of parameters
		----------------------------------------*/
		switch (dimension) {
			case 0:
				correct_num_params = (numwords == OG_MIN_NUM_PARAMS);
				break;
			case 1:
				correct_num_params = (numwords == (OG_MIN_NUM_PARAMS + 4));
				break;
			case 2:
			case 3:
				correct_num_params = (numwords == (OG_MIN_NUM_PARAMS + 6));
				break;
			default:
				p_error("Invalid dimension for observation grid",
					line);
		}

		if (!correct_num_params) {
			sprintf(error_msg,
				"Incorrect number of parameters for %d-D grid",dimension);
			p_error(error_msg,line);
		}

		/* Allocate memory for observation grid
		---------------------------------------*/
		if (first_obs_grid_E == NULL) {
			first_obs_grid_E = (obs_grid_t *)
				calloc((size_t) 1,sizeof(obs_grid_t));
			if (!first_obs_grid_E)
				p_error("Cannot allocate memory (calloc) for obs grid",
					line);
			current_obs_grid = first_obs_grid_E;
		} else {
			current_obs_grid->next = (obs_grid_t *)
				calloc((size_t) 1,sizeof(obs_grid_t));
			if (!current_obs_grid->next)
				p_error("Cannot allocate memory (calloc) for obs grid",
					line);
			current_obs_grid = current_obs_grid->next;
		}

		/* Set the observation grid dimension
		-------------------------------------*/
		current_obs_grid->dimension = dimension;

		/* Get observation grid name
		----------------------------*/
		current_obs_grid->name = (char *) malloc((size_t) 
			strlen(word[OG_NAME_POS])+1);
		if (!current_obs_grid->name)
			p_error("Cannot allocate memory for observation grid name",
			 line);
		strcpy(current_obs_grid->name,word[OG_NAME_POS]);

		/* Get the print options
		------------------------*/
		i = 0;
		current_obs_grid->print[DISPL]   = FALSE;
		current_obs_grid->print[STRAIN]  = FALSE;
		current_obs_grid->print[STRESS]  = FALSE;
		current_obs_grid->print[PSTRAIN] = FALSE;
		current_obs_grid->print[PSTRESS] = FALSE;
		while ((temp_char = word[OG_PRINT_OPS_POS][i]) != '\0') {
			switch (temp_char) {
				case DISPL_CHAR:
					current_obs_grid->print[DISPL] = TRUE;
					break;
				case STRAIN_CHAR:
					current_obs_grid->print[STRAIN] = TRUE;
					break;
				case STRESS_CHAR:
					current_obs_grid->print[STRESS] = TRUE;
					break;
				case PRINCIPAL_CHAR:
					i++;
					switch (word[OG_PRINT_OPS_POS][i]) {
						case STRAIN_CHAR:
							current_obs_grid->print[PSTRAIN] = TRUE;
							break;
						case STRESS_CHAR:
							current_obs_grid->print[PSTRESS] = TRUE;
							break;
						default:
							p_error("Invalid observation grid print option",
								line);
					}
					break;
				default:
					p_error("Invalid observation grid print option",line);
			}
			i++;
		}
			

		/* Get the input coordinate system
		----------------------------------*/
		if ((current_obs_grid->endpt_csys = 
			find_csys(word[OG_INPUT_CSYS_POS])) == NULL) {
			p_error("Undefined coordinate system",line);
		}

		/* Get the observation point coordinate system
		----------------------------------------------*/
		if ((current_obs_grid->obspt_csys = 
			find_csys(word[OG_OBSPT_CSYS_POS])) == NULL) {
			p_error("Undefined coordinate system",line);
		}

		/* Get the output coodinate system
		-----------------------------------*/
		if ((current_obs_grid->outp_csys = 
			find_csys(word[OG_DATA_CSYS_POS])) == NULL) {
			p_error("Undefined coordinate system",line);
		}

		/* Get the beginning & ending coordinates
		-----------------------------------------*/
		for (i=0; i < 3; i++) {
			current_obs_grid->begin[i] = atof(word[OG_BEGIN_POS+i]);
			if (dimension != 0) {
				current_obs_grid->end[i] = atof(word[OG_END_POS+i]);
			}
		}

		/* Get number of points along each coord axis in grid
		----------------------------------------------------*/
		num_ones = 0;
		for (i=0; i < ((dimension == 2) ? 3 : dimension); i++) {
			current_obs_grid->numpts[i] = numpts =
				atoi(word[OG_NUMPTS_POS+i]);
			if (numpts < 2) {
				if (dimension == 2 && numpts == 1) {
						num_ones++;
				} else {
					sprintf(error_msg,
						"%dD observation grid axes require %d or more points",
						dimension,((dimension == 2) ? 1 : 2));
					p_error(error_msg,line);
				}
			}
		}
		if (dimension == 2 && num_ones != 1) {
			p_error(
				"1 (& only 1) 2D observation grid axis requires 1 point",
				line);
		}

      /*             98-12-09
		   Do not convert the begin & end pts to global coords !!!!!
         So, no transformation in global coordinate system is required
		*/

      /*
		transform_position_vector(INVERSE_TRANS,
			current_obs_grid->endpt_csys->origin,
			current_obs_grid->endpt_csys->local_rot,
			current_obs_grid->begin);
		transform_position_vector(INVERSE_TRANS,
			current_obs_grid->endpt_csys->origin,
			current_obs_grid->endpt_csys->local_rot,
			current_obs_grid->end);
      */

	}

	return(0);
}


/********************* Function: read_objs_elts_verts *********************
* Reads objects, elements, and vertices from the input file, skipping blank
* and comment lines.  Stops reading when a line beginning with END_STMT is
* reached.  Objects, element and vertices are stored in seperate linked-lists.
* Pointers between these lists are set to indicate which elements belong to
* an object and which vertices belong to an element.
*****************************************************************************/
int      read_objs_elts_verts(void)
{
	int		numwords;
	char	*word[MAXWORDS];
	char	line[MAXLINE];
	vert_t *current_vert;
	obj_t *current_obj;
	elt_t *current_elt;

	/* Read in vertices and elements
	--------------------------------*/
	for (;;) {

		/* read line from input file, increment linenum
		-----------------------------------------------*/
		numwords = read_line(line,word);
		linenum_E++;

		/* skip blank and comment lines
		-------------------------------*/
		if (numwords == 0)
			continue;

		/* exit loop when end of list reached
		-------------------------------------*/ 
		if (!strcmp(word[0],END_STMT))
			break;

		/* read vertex info
		-------------------*/
		if (word[0][0] == V_CHAR) {
			get_vert_info(&current_vert,numwords,word,line);
		}

		/* Read object info
		-------------------*/
		else if (word[0][0] == OBJ_CHAR) {
			get_obj_info(&current_obj,numwords,word,line);
		}

		/* Read element info
		--------------------*/
		else if (word[0][0] == E_CHAR) {
			get_elt_info(&current_elt,current_obj,numwords,
				word,line);
		}

	}

	/* Make sure last object contains element(s)
	--------------------------------------------*/
	if (current_obj->first_elt == NULL)
		p_error("Last object contains no elements",NULL);

	return(0);
}


/*********************** Function: print_obj_titles *************************
* Prints the object name and column titles to the output (& temporary output)
* files.
*
* In:	current_obj	- the object for which titles are to be printed
*****************************************************************************/
void     print_obj_titles(obj_t *current_obj)
{
	fprintf(ofp_E,"\n\n====================================================\n");
	fprintf(ofp_E,    "              OBJECT: %s\n",current_obj->name);
	fprintf(ofp_E,    "ELT CENTER COORD SYS: %s\n",
		current_obj->pos_csys->name);
	fprintf(ofp_E,    "====================================================\n");

	/* Print titles
	---------------*/
	if (current_obj->print[DISPL]) {
		fprintf(tempfp_E[DISPL],OBJ_DISPL_TITLE);
		fprintf(tempfp_E[DISPL],OBJ_LOC_LABELS);
		fprintf(tempfp_E[DISPL],OBJ_DISPL_LABELS);
		fprintf(tempfp_E[DISPL],OBJ_BC_CSYS_LABELS);
		fprintf(tempfp_E[DISPL],OBJ_LOC_UNDLNS);
		fprintf(tempfp_E[DISPL],OBJ_DISPL_UNDLNS);
		fprintf(tempfp_E[DISPL],OBJ_BC_CSYS_UNDLNS);
	}
	if (current_obj->print[STRESS]) {
		fprintf(tempfp_E[STRESS],OBJ_STRESS_TITLE);
		fprintf(tempfp_E[STRESS],OBJ_LOC_LABELS);
		fprintf(tempfp_E[STRESS],OBJ_STRESS_LABELS);
		fprintf(tempfp_E[STRESS],OBJ_BC_CSYS_LABELS);
		fprintf(tempfp_E[STRESS],OBJ_LOC_UNDLNS);
		fprintf(tempfp_E[STRESS],OBJ_STRESS_UNDLNS);
		fprintf(tempfp_E[STRESS],OBJ_BC_CSYS_UNDLNS);
	}
}


/********************** Function: print_obs_grid_titles **********************
* Prints the observation grid name and column titles to the output
* (& temporary output) files.
*
* In:	current_obs_grid	- the obs grid for which titles are to be printed
*****************************************************************************/
void     print_obs_grid_titles(obs_grid_t *current_obs_grid)
{
	fprintf(ofp_E,"\n\n====================================================\n");
	fprintf(ofp_E,   "%d-D OBSERVATION GRID: %s\n",
		current_obs_grid->dimension, current_obs_grid->name);
	fprintf(ofp_E,    " OBS POINT COORD SYS: %s\n",
		current_obs_grid->obspt_csys->name);
	fprintf(ofp_E,    "    OUTPUT COORD SYS: %s\n",
		current_obs_grid->outp_csys->name);
	fprintf(ofp_E,    "====================================================\n");

	/* Print titles to temporary files
	----------------------------------*/
	if (current_obs_grid->print[DISPL]) {
		fprintf(tempfp_E[DISPL],OG_DISPL_TITLE);
		fprintf(tempfp_E[DISPL],OG_LOC_LABELS);
		fprintf(tempfp_E[DISPL],OG_DISPL_LABELS);
		fprintf(tempfp_E[DISPL],OG_LOC_UNDLNS);
		fprintf(tempfp_E[DISPL],OG_DISPL_UNDLNS);
	}
	if (current_obs_grid->print[STRAIN]) {
		fprintf(tempfp_E[STRAIN],OG_STRAIN_TITLE);
		fprintf(tempfp_E[STRAIN],OG_LOC_LABELS);
		fprintf(tempfp_E[STRAIN],OG_STRAIN_LABELS);
		fprintf(tempfp_E[STRAIN],OG_LOC_UNDLNS);
		fprintf(tempfp_E[STRAIN],OG_STRAIN_UNDLNS);
	}
	if (current_obs_grid->print[PSTRAIN]) {
		fprintf(tempfp_E[PSTRAIN],OG_PSTRAIN_TITLE);
		fprintf(tempfp_E[PSTRAIN],OG_LOC_LABELS);
		fprintf(tempfp_E[PSTRAIN],OG_PSTRAIN_LABELS);
		fprintf(tempfp_E[PSTRAIN],OG_LOC_UNDLNS);
		fprintf(tempfp_E[PSTRAIN],OG_PSTRAIN_UNDLNS);
	}
	if (current_obs_grid->print[STRESS]) {
		fprintf(tempfp_E[STRESS],OG_STRESS_TITLE);
		fprintf(tempfp_E[STRESS],OG_LOC_LABELS);
		fprintf(tempfp_E[STRESS],OG_STRESS_LABELS);
		fprintf(tempfp_E[STRESS],OG_LOC_UNDLNS);
		fprintf(tempfp_E[STRESS],OG_STRESS_UNDLNS);
	}
	if (current_obs_grid->print[PSTRESS]) {
		fprintf(tempfp_E[PSTRESS],OG_PSTRESS_TITLE);
		fprintf(tempfp_E[PSTRESS],OG_LOC_LABELS);
		fprintf(tempfp_E[PSTRESS],OG_PSTRESS_LABELS);
		fprintf(tempfp_E[PSTRESS],OG_LOC_UNDLNS);
		fprintf(tempfp_E[PSTRESS],OG_PSTRESS_UNDLNS);
	}

}


/************************** Function: print_elt_data ***********************
* Calculates and prints displacement and traction data for an element, given:
*
* In:	current_obj	- the object to which this element belongs
*		current_elt	- the element for which data is to be printed
*		elt_num		- the element number within the object
* Out:	stress		- the stress tensor at the element center due to ALL elts
*		displ		- the displacement of the element center due to ALL elts
*					  EXCEPT this element
****************************************************************************/
void     print_elt_data(obj_t *current_obj, elt_t *current_elt, int elt_num,double displ[3], double stress[3][3])
{
	double	half_b[3];
	double	displ_pos[3];
	double	displ_neg[3];
	double	normal_vector[3];
	double	traction[3];
	double	b[3];
	double	center[3];
	int		i;


	copy_vector(current_elt->elt_csys.origin,center);
	transform_position_vector(FORWARD_TRANS,
		current_obj->pos_csys->origin,
		current_obj->pos_csys->local_rot,
		center);

	/* Print element location info
	------------------------------*/
	for (i=0; i < NUM_PR_OPTS; i++) {
		if (current_obj->print[i]) {
			fprintf(tempfp_E[i],OBJ_LOC_FMT,elt_num,center[0],
				center[1],center[2]);
		}
	}

	/* Print displacement data if requested
	---------------------------------------*/
	if (current_obj->print[DISPL]) {
		if (below_vertex_E || near_vertex_E) {
			initialize_vector(b,null_value_E);
			initialize_vector(displ_pos,null_value_E);
			initialize_vector(displ_neg,null_value_E);
		} else {
			/* Rotate displacement vector to the bc coordinate sys
			------------------------------------------------------*/
			rotate_vector(FORWARD_ROT,current_elt->bc_csys->local_rot,displ);

			/* Compute the absolute displacements of the pos and neg sides
			   of the element by adding the displacement discontinuity to
			   the calculated displacment
			---------------------------------------------------------------*/
			for (i=0; i < 3; i++) {
				b[i] = *current_elt->b[i];
			}
			scalar_vector_mult(0.5,b,half_b);
			add_vectors(displ,half_b,displ_pos);
			subtract_vectors(displ,half_b,displ_neg);
		}

		fprintf(tempfp_E[DISPL],OBJ_DISPL_FMT,
			b[0],displ_pos[0],displ_neg[0],
			b[1],displ_pos[1],displ_neg[1],
			b[2],displ_pos[2],displ_neg[2]);
	}

	/* Print stress (traction) data if requested
	--------------------------------------------*/
	if (current_obj->print[STRESS]) {

		/* Calculate the traction vector on the element plane and
	   	rotate to bc coordinates
		---------------------------------------------------------*/
		normal_vector[0] = normal_vector[1] = 0.0;
		normal_vector[2] = -1.0;
		rotate_vector(INVERSE_ROT,current_elt->elt_csys.local_rot,
			normal_vector);
		cauchy(stress,normal_vector,traction);
		rotate_vector(FORWARD_ROT,
			current_elt->bc_csys->local_rot,traction);


		fprintf(tempfp_E[STRESS],OBJ_STRESS_FMT,traction[0],
			traction[1],traction[2]);
	}

	/* Print BC coord sys name
	--------------------------*/
	for (i=0; i < NUM_PR_OPTS; i++) {
		if (current_obj->print[i]) {
			fprintf(tempfp_E[i],OBJ_BC_CSYS_FMT,
				current_elt->bc_csys->name);
		}
	}

}


/************************ Function: print_obs_pt_data ***********************
* Prints stress, strain, and displacement data for an observation point.
*
* In:	current_obs_grid	- the obs grid to which the obs point belongs
*		x					- coordinates (global) of the observation point
*		displ				- the displacement (global coords) at the obs pt
*		strain				- the strain (global coords) at the obs point
****************************************************************************/
void     print_obs_pt_data(obs_grid_t *current_obs_grid, double x[3],double displ[3], double strain[3][3])
{
	double	stress[3][3];
	double	prin[3];
	double	traj[3][3];
	double x_copy[3];
	csys_t *obspt_csys;
	csys_t *outp_csys;
	int		i;

	obspt_csys = current_obs_grid->obspt_csys;
	outp_csys = current_obs_grid->outp_csys;

	copy_vector(x,x_copy);
	transform_position_vector(FORWARD_TRANS,obspt_csys->origin,
		obspt_csys->local_rot,x_copy);

	/* Print observation point location to temp files
	-------------------------------------------------*/
	for (i=0; i < NUM_PR_OPTS; i++) {
		if (current_obs_grid->print[i]) {
			fprintf(tempfp_E[i],OG_LOC_FMT,x_copy[0],x_copy[1],x_copy[2]);
		}
	}

	/* Print displacement data to temp file
	---------------------------------------*/
	if (current_obs_grid->print[DISPL]) {
		rotate_vector(FORWARD_ROT,outp_csys->local_rot,displ);
		if (below_vertex_E || near_vertex_E)
			initialize_vector(displ,null_value_E);
		fprintf(tempfp_E[DISPL],OG_DISPL_FMT,displ[0],displ[1],displ[2]);
	}

	/* Print stress data to temp file
	---------------------------------*/
	if (current_obs_grid->print[STRESS] || current_obs_grid->print[PSTRESS]) {

		strain_to_stress(strain, shear_mod_E, lame_lambda_E, stress);
		rotate_tensor(FORWARD_ROT,outp_csys->local_rot,stress);

		if (current_obs_grid->print[STRESS]) {
			if (below_vertex_E || near_vertex_E)
				initialize_matrix(stress,null_value_E);
			fprintf(tempfp_E[STRESS],OG_STRESS_FMT,stress[0][0],stress[1][1],
				stress[2][2], stress[0][1], stress[1][2], stress[0][2]);
		}

		if (current_obs_grid->print[PSTRESS]) {
			/* Calculate principal stresses
			-------------------------------*/
			if (below_vertex_E || near_vertex_E) {
				initialize_vector(prin,null_value_E);
				initialize_matrix(traj,null_value_E);
			} else {
				principal(stress, prin, traj);
			}
			fprintf(tempfp_E[PSTRESS],OG_PSTRESS_FMT,
				traj[0][0], traj[0][1], traj[0][2], prin[0],
				traj[1][0], traj[1][1], traj[1][2], prin[1],
				traj[2][0], traj[2][1], traj[2][2], prin[2]);
		}
	}


	/* Print strain data to temp file
	---------------------------------*/
	if (current_obs_grid->print[STRAIN] || current_obs_grid->print[PSTRAIN]) {

		rotate_tensor(FORWARD_ROT,outp_csys->local_rot,strain);

		if (current_obs_grid->print[STRAIN]) {
			if (below_vertex_E || near_vertex_E)
				initialize_matrix(strain,null_value_E);
			fprintf(tempfp_E[STRAIN],OG_STRAIN_FMT,strain[0][0],strain[1][1],
				strain[2][2], strain[0][1], strain[1][2], strain[0][2]);
		}

		if (current_obs_grid->print[PSTRAIN]) {
			/* Calculate principal strains
			------------------------------*/
			if (below_vertex_E || near_vertex_E) {
				initialize_vector(prin,null_value_E);
				initialize_matrix(traj,null_value_E);
			} else {
				principal(strain, prin, traj);
			}
			fprintf(tempfp_E[PSTRAIN],OG_PSTRAIN_FMT,
				traj[0][0], traj[0][1], traj[0][2], prin[0],
				traj[1][0], traj[1][1], traj[1][2], prin[1],
				traj[2][0], traj[2][1], traj[2][2], prin[2]);
		}
	}

	below_vertex_E = FALSE;

   /* NEW: 98-12-09 */
   near_vertex_E  = FALSE;
}


/******************** Function: determine_burgers_vectors ********************
* This function was re-written by Scott T. Marshall with the assitance 
* of the Google Gemini (LLM). The purpose was to parallelize the matrix 
* building and inversion, which now uses the efficient and parallelized 
* LAPACK and OpenBLAS libraries. Condition numbers may now be lower than calculated 
* by the old Numerical Recipes routines in nr.c
* Results have been validated for a wide range of models and obs grids.
*-----------------------------------------------------------------------------
* Solves for the unknown Burgers vector components.
* 1. Solver: LAPACK dgetrf (LU Factorization) -> dgetrs (Solve)
* 2. Condition Number: LAPACK dgecon using 'I' (Infinity Norm / Max Row Sum)
* - Matches legacy 'array_max_norm' definition.
* 3. Sign Convention: Influence coefficients *-1 to match poly3d sign convention.
* 4. Safety: Checks for 0 unknowns (Kinematic models) to prevent segfaults.
*****************************************************************************/
void determine_burgers_vectors(void)
{
    int     i, k;
    long    num_unknowns = 0;
    long    matrix_size;
    double  *flat_matrix = NULL; 
    double  *rhs_vector = NULL;  
    int     *ipiv = NULL;        
    int     info;                
    long    n_rhs = 1;           
    
    /* Variables for Condition Number Estimation */
    double  anorm = 0.0;         
    double  rcond;               
    double  *work = NULL;        
    int     *iwork = NULL;       
    
    elt_t   *elt1, *elt2;
    elt_t   **elt_array;         
    long    *row_offsets;        
    long    num_elements = 0;    
    
    double  mu;                  
    time_t  start_asm, end_asm; 
	
	/* RAM Calculation Var */
    double matrix_ram_mb;
	
    /* Calculate Shear Modulus locally */
    mu = youngs_mod_E / (2.0 * (1.0 + psn_ratio_E));

    /* -----------------------------------------------------------------------
       STEP 1: Count elements and allocate index arrays
       ----------------------------------------------------------------------- */
    elt1 = first_elt_E;
    while (elt1 != NULL) {
        num_elements++;
        elt1 = elt1->next;
    }

    if (num_elements == 0) return;

    elt_array   = (elt_t **) malloc(num_elements * sizeof(elt_t *));
    row_offsets = (long *)   malloc(num_elements * sizeof(long));
    
    if (!elt_array || !row_offsets) 
        p_error("Memory allocation failed in determine_burgers_vectors", NULL);

    /* -----------------------------------------------------------------------
       STEP 2: Map Unknowns and Setup 'b' Pointers
       ----------------------------------------------------------------------- */
    k = 0;
    elt1 = first_elt_E;
    while (elt1 != NULL) {
        elt_array[k] = elt1;
        row_offsets[k] = -1; 

        for (i = 0; i < 3; i++) {
            if (elt1->bc_type[i] == TRACTION_BC) {
                if (row_offsets[k] == -1) row_offsets[k] = num_unknowns;
                num_unknowns++;
            } else {
                /* BVECTOR_BC: Known value */
                elt1->b[i] = &(elt1->bc[i]);
            }
        }
        elt1 = elt1->next;
        k++;
    }

    /* -----------------------------------------------------------------------
       STEP 3: Handle Kinematic Models (All BCs are 'b')
       ----------------------------------------------------------------------- */
    if (num_unknowns == 0) {
        printf("No traction BCs found (Kinematic model). Skipping solver.\n");
        cond_num_E = -1.0; 
        free(elt_array);
        free(row_offsets);
        return; 
    } else {
		int num_threads = 1;
        #ifdef _OPENMP
            num_threads = omp_get_max_threads();
            printf("OpenMP enabled       : %d Threads\n", num_threads);
        #else
            printf("OpenMP disabled      : 1 Thread\n");
        #endif
		
		/* Calculate RAM: num_eqns^2 * 8 bytes (double) */
        matrix_ram_mb = num_unknowns * num_unknowns * sizeof(double) / (1024 * 1024);
        
		/*print some useful info to stdout*/
        printf("IC matrix dimensions : %ld x %ld\n", num_unknowns, num_unknowns);
        /*print in MB or GB depending on size*/
		if (matrix_ram_mb < 1000)  printf("IC matrix RAM usage  : %.2f MB\n", matrix_ram_mb);
		else                       printf("IC matrix RAM usage  : %.2f GB\n", matrix_ram_mb/1000);

        
        fflush(stdout);
	}

    /* -----------------------------------------------------------------------
       STEP 4: Allocate Matrix and Solver Arrays
       ----------------------------------------------------------------------- */
    if (b_vector_E) free(b_vector_E); 
    b_vector_E = (double *) calloc(num_unknowns, sizeof(double));
    if (!b_vector_E) p_error("Cannot allocate b_vector_E", NULL);

    long curr_idx = 0;
    for (k = 0; k < num_elements; k++) {
        elt1 = elt_array[k];
        for (i = 0; i < 3; i++) {
            if (elt1->bc_type[i] == TRACTION_BC) {
                elt1->b[i] = &b_vector_E[curr_idx];
                curr_idx++;
            }
        }
    }

    matrix_size = num_unknowns * num_unknowns;
    flat_matrix = (double *) calloc(matrix_size, sizeof(double));
    rhs_vector  = (double *) calloc(num_unknowns, sizeof(double));
    ipiv        = (int *)    calloc(num_unknowns, sizeof(int));
    
    /* Workspace for dgecon (4*N doubles, N ints) */
    work        = (double *) calloc(4 * num_unknowns, sizeof(double));
    iwork       = (int *)    calloc(num_unknowns, sizeof(int));

    if (!flat_matrix || !rhs_vector || !ipiv || !work || !iwork) 
        p_error("Memory allocation failed for matrix/vectors", NULL);

    /* -----------------------------------------------------------------------
       STEP 5: Assembly (Build Matrix and RHS)
       ----------------------------------------------------------------------- */
    printf("IC matrix building   : ");
	fflush(stdout);
    time(&start_asm);

#pragma omp parallel for default(none) \
    shared(num_elements, elt_array, row_offsets, flat_matrix, rhs_vector, num_unknowns, mu, lame_lambda_E) \
    private(k, i, elt1, elt2) \
    schedule(dynamic)
    for (k = 0; k < num_elements; k++) {
        
        elt1 = elt_array[k];
        long row_start_idx = row_offsets[k];

        if (row_start_idx == -1) continue;

        double stress_ic[3][3][3];
        double traction_ic[3][3]; 
        double dummy_displ[3][3];
        double normal_global[3];
        double traction_global[3];
        double *bc_origin;
        double (*bc_rot)[3];
        int t_comp, b_comp;

        /* Normal in Global Coordinates */
        normal_global[0] = 0.0; normal_global[1] = 0.0; normal_global[2] = 1.0;
        rotate_vector(INVERSE_ROT, elt1->elt_csys.local_rot, normal_global);

        /* Determine which Coordinate System the BCs are defined in */
        if (elt1->bc_csys != NULL) {
            bc_origin = elt1->bc_csys->origin;
            bc_rot    = elt1->bc_csys->local_rot;
        } else {
            bc_origin = elt1->elt_csys.origin;
            bc_rot    = elt1->elt_csys.local_rot;
        }

        long src_k;
        for (src_k = 0; src_k < num_elements; src_k++) {
            elt2 = elt_array[src_k];

            /* Calculate Stress Influence (Global) */
            displ_strain_ics_poly_elt(FALSE, TRUE, elt2, bc_origin, 
                                      dummy_displ, stress_ic, NULL);

            /* Convert Stress IC to Traction IC in BC Coordinates */
                    for (b_comp = 0; b_comp < 3; b_comp++) {
                        
                        double temp_stress_ic[3][3];
                        double trace;
                        int i_s, j_s;
                        
                        trace = stress_ic[b_comp][0][0] + stress_ic[b_comp][1][1] + stress_ic[b_comp][2][2];
            
                        for (i_s=0; i_s < 3; i_s++) {
                            for (j_s=0; j_s < 3; j_s++) {
                                temp_stress_ic[i_s][j_s] = 2*mu*stress_ic[b_comp][i_s][j_s];
                            }
                            temp_stress_ic[i_s][i_s] += lame_lambda_E*trace;
                        }
            
                        for (i_s=0; i_s < 3; i_s++) {
                            traction_global[i_s] = 0.0;
                            for (j_s=0; j_s < 3; j_s++) {
                                traction_global[i_s] += temp_stress_ic[i_s][j_s] * normal_global[j_s];
                            }
                        }
            
                        // Apply sign correction
                        for (i_s=0; i_s<3; i_s++) {
                            traction_global[i_s] *= -1.0;
                        }
            
                        /* Rotate Traction to BC System (Global -> Local) */
                        rotate_vector(FORWARD_ROT, bc_rot, traction_global);
            
                        traction_ic[0][b_comp] = traction_global[0];
                        traction_ic[1][b_comp] = traction_global[1];
                        traction_ic[2][b_comp] = traction_global[2];
                    }
            /* Distribute to Matrix or RHS */
            long current_row = row_start_idx;
            
            for (t_comp = 0; t_comp < 3; t_comp++) {
                if (elt1->bc_type[t_comp] == BVECTOR_BC) continue;
                
                long src_col_base = row_offsets[src_k];
                long current_src_col = src_col_base; 

                for (b_comp = 0; b_comp < 3; b_comp++) {
                    double influence = traction_ic[t_comp][b_comp];

                    if (elt2->bc_type[b_comp] == TRACTION_BC) {
                        /* Add to Matrix (Column-Major: A[row + col*N]) */
                        flat_matrix[current_row + current_src_col * num_unknowns] += influence;
                        current_src_col++;
                    } else {
                        /* Add to RHS */
                        double known_val = *(elt2->b[b_comp]);
                        rhs_vector[current_row] -= influence * known_val;
                    }
                }
                current_row++;
            }
        }
        
        /* Add Prescribed Traction to RHS */
        long r_idx = row_start_idx;
        for (t_comp = 0; t_comp < 3; t_comp++) {
            if (elt1->bc_type[t_comp] == TRACTION_BC) {
                rhs_vector[r_idx] += elt1->bc[t_comp]; 
                r_idx++;
            }
        }
    }
    
    time(&end_asm);
    /*printf("%.0f seconds\n", difftime(end_asm, start_asm));*/
	printf("Done\n");

    /* -----------------------------------------------------------------------
       STEP 6: Solve System (Factored + Condition + Solve)
       ----------------------------------------------------------------------- */
    printf("IC matrix solving    : ");
    fflush(stdout);

    /* 6a. Calculate Infinity-Norm (Max Row Sum) */
    /* This matches the logic of 'array_max_norm' in original poly3d.c */
    /* Note: flat_matrix is Column-Major. Element (row, col) is at index [row + col*N] */
    anorm = 0.0;
    for (int r = 0; r < num_unknowns; r++) {
        double row_sum = 0.0;
        for (int c = 0; c < num_unknowns; c++) {
            row_sum += fabs(flat_matrix[r + c * num_unknowns]);
        }
        if (row_sum > anorm) anorm = row_sum;
    }

    int lapack_N    = (int)num_unknowns;
    int lapack_NRHS = (int)n_rhs;
    int lapack_LDA  = (int)num_unknowns;
    int lapack_LDB  = (int)num_unknowns;

    /* 6b. LU Factorization */
    dgetrf_(&lapack_N, &lapack_N, flat_matrix, &lapack_LDA, ipiv, &info);

    if (info != 0) {
        if (info < 0) printf("LAPACK dgetrf Error: Argument %d illegal\n", -info);
        else          printf("LAPACK dgetrf Error: Singular matrix (U[%d,%d]=0)\n", info, info);
        p_error("Factorization failed.", NULL);
    }

    /* 6c. Condition Number Estimation (Infinity Norm) */
    char norm_type = 'I'; /* 'I' for Infinity Norm (Max Row Sum) */
    dgecon_(&norm_type, &lapack_N, flat_matrix, &lapack_LDA, &anorm, &rcond, work, iwork, &info);
    
    /* Store condition number in global variable */
    if (rcond > 0.0) cond_num_E = 1.0 / rcond;
    else             cond_num_E = 1.0e+30; 

    /* 6d. Solve */
    char trans_type = 'N'; /* No transpose */
    dgetrs_(&trans_type, &lapack_N, &lapack_NRHS, flat_matrix, &lapack_LDA, ipiv, rhs_vector, &lapack_LDB, &info);

    if (info != 0) {
        printf("LAPACK dgetrs Error: Info = %d\n", info);
        p_error("Back-substitution failed.", NULL);
    }
    
    printf("Done\n");

    /* -----------------------------------------------------------------------
       STEP 7: Update Solution
       ----------------------------------------------------------------------- */
    for (i = 0; i < num_unknowns; i++) {
        b_vector_E[i] = rhs_vector[i];
    }

    free(elt_array);
    free(row_offsets);
    free(flat_matrix);
    free(rhs_vector); 
    free(ipiv);
    free(work);
    free(iwork);

    return;
}


/*************************** Function: read_line ***************************
* Uses getwords() to read a line from the input file. Adjusts numwords
* appropriately if the line contains a comment character.  Returns number
* of words on line.
*
* Out:	line	- the line read from the input file
* In:	word[i]	- the ith word on the line (null-terminated string)
***************************************************************************/
int      read_line(char *line, char *word[])
{
	int		numwords;
	/*char	error_msg[MAX_ERROR_MSG];*/
	int		i, j;
	int		exit;

	/* get line, exit function on EOF
	---------------------------------*/
	if ((numwords = getwords(ifp_E,line,MAXLINE,word,MAXWORDS,
		CONTINUE_CHAR)) < 0) {
		switch (numwords) {
			case GW_EOF_ERR:
				p_error("Unexpected EOF in getwords()\n",NULL);
			case GW_MALLOC_ERR:
				p_error("Memory allocation error in getwords()\n",NULL);
			case GW_MAXWORDS_ERR:
				p_error("Too many words error in getwords()\n",NULL);
		}
	} /*if*/

	/* if line contains a comment, adjust numwords accordingly
	----------------------------------------------------------*/
	exit = FALSE;
	for (i=0; !exit && (i < numwords); i++) {

		/* Loop over characters in word.  If comment character is found,
		   replace with '\0' and exit loop.
		----------------------------------------------------------------*/
		j = 0;
		while (word[i][j] != '\0') {
			if (word[i][j] == COMMENT_CHAR) {
				word[i][j] = '\0';
				exit = TRUE;
				break;
			}
			j++;
		}
	}
	if (exit) 
		numwords = (j == 0) ? i-1 : i;

	return(numwords);
}


/************************* Function: get_double_var **************************
* Function for assigning values to double variables.  Returns TRUE if an
* assignment is made, or FALSE otherwise.
*
* In:		var_name	- name used for this variable in the input file
*			word		- array of words given in the input line
*			numwords	- number of words on the input line
* In/Out:	var			- variable to which a value is to be assigned
*****************************************************************************/
int      get_double_var(double *var, const char *var_name, const char *const word[], int numwords)
{
	if (!strcmp(word[CONST_NAME_POS],var_name)) {
		if (numwords == CONST_NUM_PARAMS) {
			*var = atof(word[CONST_VALUE_POS]);
		}
		return(TRUE);
	}
	return(FALSE);
}


/************************* Function: get_boolean_var *************************
* Function for assigning values to boolean variables.  Returns TRUE if an
* assignment is made, or FALSE otherwise.
*
* In:		var_name		- name used for this variable in the input file
*			true_string		- true value string for this var in input file
*			false_string	- false value string for this var in input file
*			word			- array of words given in the input line
*			numwords		- number of words on the input line
*			line			- line read from the input file
* In/Out:	var				- variable to which a value is to be assigned
*****************************************************************************/
int      get_boolean_var(int *var, const char *var_name, const char *true_string,
		const char *false_string, const char *const word[], int numwords,
		const char *line)
{
	char	error_msg[MAX_ERROR_MSG];

	if (!strcmp(word[CONST_NAME_POS],var_name)) {
		if (numwords == CONST_NUM_PARAMS) {
			if (!strcmp(word[CONST_VALUE_POS],true_string))
				*var = TRUE;
			else if (!strcmp(word[CONST_VALUE_POS],false_string))
				*var = FALSE;
			else {
				sprintf(error_msg, "%s requires a value of \"%s\" or \"%s\"",
					var_name, true_string, false_string);
				p_error(error_msg,line);
			}
		}
		return(TRUE);
	}
	return(FALSE);
}


/************************** Function: get_text_var ***************************
* Function for assigning values to test variables.  Returns TRUE if an
* assignment is made, or FALSE otherwise.
*
* In:		var_name	- name used for this variable in the input file
*			word		- array of words given in the input line
*			numwords	- number of words on the input line
*			line		- line read from the input file
* In/Out:	var			- variable to which a value is to be assigned
*****************************************************************************/
int      get_text_var(char **var, const char *var_name, const char *const word[],
		int numwords, const char *line)
{
	char	error_msg[MAX_ERROR_MSG];
	
	if (!strcmp(word[CONST_NAME_POS],var_name)) {
		if (numwords == CONST_NUM_PARAMS) {
			*var = (char *) malloc((size_t) strlen(word[CONST_VALUE_POS])+1);
			if (!*var) {
				sprintf(error_msg,"Cannot allocate memory for %s",var_name);
				p_error(error_msg,line);
			}
			strcpy(*var,word[CONST_VALUE_POS]);
		}
		return(TRUE);
	}
	return(FALSE);
}


/************************* Function: get_vert_info *************************
* Reads vertex info and sets up new member in linked list of vertices.
*
* In:		numwords		- number of words on input line
*			word			- array of words given in the input line
*			line			- the input line
* In/Out:	current_vert	- the current (last) vertex in the linked list
****************************************************************************/
void     get_vert_info(vert_t **current_vert, int numwords, char *word[],char *line)
{
	int		i;
	char	error_msg[MAX_ERROR_MSG];

	/* Check for proper number of parameters
	----------------------------------------*/
	if (numwords != V_NUM_PARAMS) {
		sprintf(error_msg,
			"Too %s parameters specified to define vertex",
			(numwords < V_NUM_PARAMS) ? "few" : "many");
		p_error(error_msg,line);
	}

	/* Allocate memory for vertex
	-----------------------------*/
	if (first_vert_E == NULL) {
		first_vert_E = (vert_t *) calloc((size_t) 1,sizeof(vert_t)); 
		if (!first_vert_E)
			p_error("Cannot allocate memory (calloc) for vertex",
				line);
		*current_vert = first_vert_E;
	} else {
		(*current_vert)->next = (vert_t *)
			calloc((size_t) 1,sizeof(vert_t));
		if (!(*current_vert)->next)
			p_error("Cannot allocate memory (calloc) for vertex",
				line);
		*current_vert = (*current_vert)->next;
	}

	/* Get vertex name
	------------------*/
	(*current_vert)->name = (char *) malloc((size_t) 
		strlen(word[V_NAME_POS])+1);
	if (!(*current_vert)->name)
		p_error("Cannot allocate memory for vertex name",
		 line);
	strcpy((*current_vert)->name,word[V_NAME_POS]);

	/* Get the coodinate system
	----------------------------*/
	if (((*current_vert)->csys = find_csys(word[V_CSYS_POS]))
		== NULL) {
		p_error("Undefined coordinate system",
			line);
	}

	/* Read the vertex position
	---------------------------*/
	for (i=0; i < 3; i++)
		(*current_vert)->x[i] = atof(word[V_X_POS+i]);

	/* Transform vertex position vector to global coords
	----------------------------------------------------*/
	transform_position_vector(INVERSE_TRANS,
		(*current_vert)->csys->origin,
		(*current_vert)->csys->local_rot,
		(*current_vert)->x);
}


/************************* Function: get_obj_info ***************************
* Reads object info and sets up new member in linked list of objects.
*
* In:		numwords		- number of words on input line
*			word			- array of words given in the input line
*			line			- the input line
* In/Out:	current_obj		- the current (last) object in the linked list
*****************************************************************************/
void     get_obj_info(obj_t **current_obj, int numwords, char *word[], char *line)
{
	int		i;
	char	temp_char;

	/* Check for proper number of parameters
	----------------------------------------*/
	if (numwords != OBJ_MIN_NUM_PARAMS &&
		numwords != OBJ_MIN_NUM_PARAMS+2) {
		p_error(
			"Incorrect number of parameters specified to define object",
			line);
	}

	/* Allocate memory for object
	-----------------------------*/
	if (first_obj_E == NULL) {
		first_obj_E = (obj_t *) calloc((size_t) 1,sizeof(obj_t));
		if (!first_obj_E)
			p_error("Cannot allocate memory (calloc) for object",
				line);
		*current_obj = first_obj_E;
	} else {
		(*current_obj)->next = (obj_t *)
			calloc((size_t) 1,sizeof(obj_t));
		if (!(*current_obj)->next)
			p_error("Cannot allocate memory (calloc) for object",
				line);
		*current_obj = (*current_obj)->next;
	}

	/* Get object name
	------------------*/
	(*current_obj)->name = (char *) malloc((size_t) 
		strlen(word[OBJ_NAME_POS])+1);
	if (!(*current_obj)->name)
		p_error("Cannot allocate memory for object name",
		line);
	strcpy((*current_obj)->name,word[V_NAME_POS]);

	if (numwords > OBJ_MIN_NUM_PARAMS) {

		/* Get the print options
		------------------------*/
		i = 0;
		(*current_obj)->print[DISPL] = 
			(*current_obj)->print[STRESS] = FALSE;
		while ((temp_char = word[OBJ_PRINT_OPS_POS][i]) != '\0') {
			switch (temp_char) {
				case BVECTOR_CHAR:
					(*current_obj)->print[DISPL] = TRUE;
					break;
				case TRACTION_CHAR:
					(*current_obj)->print[STRESS] = TRUE;
					break;
				default:
					p_error("Invalid object print option",line);
			}
			i++;
		}

		/* Get the coordinate system for element positions
		--------------------------------------------------*/
		if (((*current_obj)->pos_csys = 
			find_csys(word[OBJ_POS_CSYS_POS])) == NULL) {
			p_error("Undefined coordinate system",line);
		}

	}

	(*current_obj)->first_elt = NULL;
}
	

/************************* Function: get_elt_info ************************
* Reads element info and sets up new member in linked list of elements.
*
* In:		numwords		- number of words on input line
*			word			- array of words given in the input line
*			line			- the input line
* In/Out:	current_elt		- the current (last) element in the linked list
*			current_obj		- the object to which current_elt belongs
*****************************************************************************/
void     get_elt_info(elt_t **current_elt, obj_t *current_obj, int numwords,char *word[], char *line)
{
	int		num_vertices;
	int		num_params;
	int		i, j;
	char	error_msg[MAX_ERROR_MSG];
	vert_t	 *vert1;
	vert_t	 *vert2;
	double  dx[3];
	double	trend;
	double	plunge;
	double	rot2[3][3];
	double	rot1[3][3];
	double	trac_bc_adjust[3];
	int		seg;
	vert_t	*vert[3];
	double	vert_x[3];
	double		vector1[3];
	double		vector2[3];
	double	normal_vector[3];
	double		x[3][3];
	double		global_x[3][3];
	disloc_seg_t	*disloc_seg;
	static char	*elt_csys_name;


	/* Increment num_elts_E
	-----------------------*/
	num_elts_E++;

	/* Get the number of vertices
	-----------------------------*/
	num_vertices = atoi(word[E_NUM_VERT_POS]);
	if (num_vertices < 3) 
		p_error("Element must have at least three vertices",line);

	/* Check for proper number of parameters
	----------------------------------------*/
	num_params = E_MIN_NUM_PARAMS+(num_vertices-3);
	if (numwords != num_params) {
		sprintf(error_msg,
			"Too %s parameters specified to define %d-sided element",
			((numwords < num_params) ? "few" : "many"),num_vertices);
		p_error(error_msg,line);
	}

	/* Allocate memory for this element
	-----------------------------------*/
	if (first_elt_E == NULL) {
		first_elt_E = (elt_t *) malloc(sizeof(elt_t));
		if (!first_elt_E)
			p_error("Cannot allocate memory for element",line);
		*current_elt = first_elt_E;
		(*current_elt)->next = NULL;
	} else {
		(*current_elt)->next = (elt_t *) malloc(sizeof(elt_t));
		if (!(*current_elt)->next)
			p_error("Cannot allocate memory for element",line);
		*current_elt = (*current_elt)->next;
		(*current_elt)->next = NULL;
	}

	/* Set object pointers to first and last elements
	-------------------------------------------------*/
	if (first_obj_E == NULL)
		p_error("No objects defined. Element must be part of an object",line);
	if (current_obj->first_elt == NULL)
		current_obj->first_elt = *current_elt;
	current_obj->last_elt = *current_elt;

	/* Set the number of vertices
	-----------------------------*/
	(*current_elt)->num_vertices = num_vertices;

	/* Allocate memory for dislocation segment array
	------------------------------------------------*/
	(*current_elt)->disloc_seg = (disloc_seg_t *)
		calloc((size_t) num_vertices, sizeof(disloc_seg_t));
	if (!(*current_elt)->disloc_seg)
		p_error("Cannot allocate memory for dislocation segment array",
			line);

	/* Get dislocation segment vertices
	-----------------------------------*/
	for (i = 0; i < num_vertices; i++) {
		j = (i == (num_vertices-1)) ? 0 : i+1;
		if (i != 0) {
			(*current_elt)->disloc_seg[i].vert[0] =
			(*current_elt)->disloc_seg[i-1].vert[1];
		} else {
			if (((*current_elt)->disloc_seg[i].vert[0] =
				find_vert(word[E_VERTEX_POS+i])) == NULL) {
				p_error("Undefined vertex",line);
			}
		}
		if (((*current_elt)->disloc_seg[i].vert[1] =
			find_vert(word[E_VERTEX_POS+j])) == NULL) {
			p_error("Undefined vertex",line);
		}
	}


	/* Inititalize rotation matrices
	--------------------------------*/
	initialize_matrix(rot2,0.0);
	initialize_matrix(rot1,0.0);

	initialize_vector((*current_elt)->elt_csys.origin,0.0);

	/* Loop over the dislocation segments
	-------------------------------------*/
	disloc_seg = (*current_elt)->disloc_seg;
	for (seg = 0; seg < (*current_elt)->num_vertices; seg++) {

		/* Calculate this dislocation segment's first vertex's contribution
		   to the element center
		-------------------------------------------------------------------*/
		for (i=0; i < 3; i++) {
			(*current_elt)->elt_csys.origin[i] += 
				disloc_seg[seg].vert[0]->x[i] / (*current_elt)->num_vertices;
		}

		/* Determine the two vertices for this dislocation segment
		----------------------------------------------------------*/
		vert1 = disloc_seg[seg].vert[0];
		vert2 = disloc_seg[seg].vert[1];

		/* Calculate trend and plunge of this dislocation segment
		--------------------------------------------------------*/
		subtract_vectors(vert2->x,vert1->x,dx);
		trend = PI/2.0 - safe_atan2(dx[1],dx[0]);
		disloc_seg[seg].trend = trend;
		plunge = -safe_atan(dx[2],sqrt(dx[0]*dx[0] + dx[1]*dx[1]));
		disloc_seg[seg].plunge = plunge;

		/* Calculate the segment-local (Comninou & Dunders) to global
		   coordinates rotation matrix
		------------------------------------------------------------*/
		rot2[0][0] = 1.0;
		rot2[1][1] = rot2[2][2] = -1.0;
		rot1[0][0] = rot1[1][1] = sin(trend);
		rot1[1][0] = -(rot1[0][1] = cos(trend));
		rot1[2][2] = 1.0;
		matrix_mult(rot2,rot1,disloc_seg[seg].local_rot);

	} /*for*/

	/* Compute element's local coordinate system
	--------------------------------------------*/
	if (elt_csys_name == NULL) {
		elt_csys_name = (char *) malloc((size_t) strlen(ELT_CSYS_NAME)+1);
		if (!elt_csys_name)
			p_error("Cannot allocate memory for elt-local coord sys name",
				NULL);
		strcpy(elt_csys_name,ELT_CSYS_NAME);
	}
	(*current_elt)->elt_csys.name = elt_csys_name;

	for (i=0; i < 3; i++) {
		for (j=0; j < 3; j++) {
			global_x[i][j] = (i == j) ? 1.0 : 0.0;
		}
	}
	for (i=0; i < 3; i++) {
		vert[i] = disloc_seg[(i*(*current_elt)->num_vertices)/3].vert[0];
	}
	subtract_vectors(vert[1]->x,vert[0]->x,vector1);
	subtract_vectors(vert[2]->x,vert[0]->x,vector2);
	normalize_vector(vector1);
	normalize_vector(vector2);

	cross_product(vector1,vector2,x[2]);
	if (vector_magnitude(x[2]) < TINY_ANGLE)
		p_error(
			"Cannot calc element normal. Elt must have a very odd shape.",
			line);
	normalize_vector(x[2]);
	cross_product(global_x[2],x[2],x[1]);
	if (vector_magnitude(x[1]) < TINY_ANGLE)
		copy_vector(global_x[1],x[1]);
	normalize_vector(x[1]);
	cross_product(x[1],x[2],x[0]);
	normalize_vector(x[0]);

	for (i=0; i < 3; i++) {
		for (j=0; j < 3; j++) {
			(*current_elt)->elt_csys.local_rot[i][j] =
			dot_product(x[i],global_x[j]);
		}
	}

	/* Check that all vertices are co-planar
	----------------------------------------*/
	for (seg=0; seg < num_vertices; seg++) {
		copy_vector((*current_elt)->disloc_seg[seg].vert[0]->x,vert_x);
		transform_position_vector(FORWARD_TRANS,
			(*current_elt)->elt_csys.origin,
			(*current_elt)->elt_csys.local_rot,vert_x);
		if (fabs(vert_x[2]) >
			fabs(sqrt(vert_x[0]*vert_x[0]+vert_x[1]*vert_x[1])/COPLANAR_LIMIT))
			p_error("Vertices are not co-planar",line);
	}

	/* Get the bc coodinate system
	------------------------------*/
	if (!strcmp(word[E_BC_CSYS_POS],ELT_CSYS_NAME)) {
		(*current_elt)->bc_csys = &((*current_elt)->elt_csys);
	} else if (((*current_elt)->bc_csys =
		find_csys(word[E_BC_CSYS_POS])) == NULL) {
		p_error("Undefined coordinate system", line);
	}

	/* Read the boundary condition types
	------------------------------------*/
	for (i=0; i < 3; i++) {
		if (word[E_BC_TYPE_POS][i] == BVECTOR_CHAR)
			(*current_elt)->bc_type[i] = BVECTOR_BC;
		else if (word[E_BC_TYPE_POS][i] == TRACTION_CHAR)
			(*current_elt)->bc_type[i] = TRACTION_BC;
		else
			p_error("Invalid boundary condition type",line);
	}
	
	/* Calculate adjustment to traction BCs due to remote stresses
	--------------------------------------------------------------*/
	scalar_vector_mult(-1.0,x[2],normal_vector);
	cauchy(rem_stress_E,normal_vector,trac_bc_adjust);
	rotate_vector(FORWARD_ROT,(*current_elt)->bc_csys->local_rot,trac_bc_adjust);

	/* Read the boundary condition values
	-------------------------------------*/
	for (i=0; i < 3; i++) {
		(*current_elt)->bc[i] = atof(word[E_BC_POS+i]);

		/* Adjust traction BC components for remote stress
		--------------------------------------------------*/
		if ((*current_elt)->bc_type[i] == TRACTION_BC)
			(*current_elt)->bc[i] -= trac_bc_adjust[i];
	}

	/* Calculate the projection of each unit component of the element
	   burgers vector into a segment-local coordinates burgers vector
	   (for calculating influence coefficients)
	-----------------------------------------------------------------*/
	for (seg = 0; seg < (*current_elt)->num_vertices; seg++) {
		for (i=0; i < 3; i++) {
			initialize_vector(disloc_seg[seg].elt_b[i],0.0);
			disloc_seg[seg].elt_b[i][i] = -1.0;
			rotate_vector(INVERSE_ROT,
				(*current_elt)->bc_csys->local_rot,
				disloc_seg[seg].elt_b[i]);
			rotate_vector(FORWARD_ROT,disloc_seg[seg].local_rot,
				disloc_seg[seg].elt_b[i]);
		}
	}
}


/************************* Function: open_temp_files *************************
* Opens the temporary output files for displacement, strain, principal strain,
* stress, and principal stress data.
*
* In:	print	- array of flags giving the data to be printed (determines
*				  which temp files need to be opened)
*****************************************************************************/
void     open_temp_files(int print[])
{
	int 	i;

	/* Open temporary files
	-----------------------*/
	for (i=0; i < NUM_PR_OPTS; i++) {
		tempfp_E[i] = NULL;
		if (print[i]) {
			if ((tempfp_E[i] = tmpfile()) == NULL) {
				p_error("Cannot open temporary file",NULL);
			} /*if*/
		} /*if*/
	} 

}


/************************ Function: close_temp_files ************************
* Closes the temporary output files for displacement, strain, principal strain,
* stress, and principal stress data.
*****************************************************************************/
void     close_temp_files(void)
{
	int		i;

	/* Close temporary files
	-----------------------*/
	for (i=0; i < NUM_PR_OPTS; i++) {
		if (tempfp_E[i] != NULL) {
			if (fclose(tempfp_E[i]) == EOF) {
				p_error("Error closing temporary file",NULL);
			} /*if*/
			tempfp_E[i] = NULL;
		} /*if*/
	}

}


/********************** Function: copy_temp_files ***************************
* Copues the temporary output files for displacement, strain, principal strain,
* stress, and principal stress data to the main output file.
*****************************************************************************/
void     copy_temp_files(void)
{
		int		i;
		FILE	*tfp;
		int		c;

		/* Rewind temp files and copy to main output file
		-------------------------------------------------*/
		for (i=0; i < NUM_PR_OPTS; i++) {
			if ((tfp = tempfp_E[i]) != NULL) {
				rewind(tfp);
				while ((c = fgetc(tfp)) != EOF)
					fputc(c,ofp_E);
			}
		}
}


/************************** Function: print_elt_geometry *********************
* Loops over all objects, printing the geometry of each element (name &
* coordinates of each vertex) to the output file. 
*****************************************************************************/
void     print_elt_geometry(void)
{
	csys_t	*elt_geom_csys;
	obj_t	*current_obj;
	elt_t	*current_elt;
	int		elt_num=0;
	int		i;
	double	x[3];
	double	first_vert_x[3];

	/* Get element geometry coordinate system
	-----------------------------------------*/
	if (elt_geom_csys_name_E == NULL) {
		if ((elt_geom_csys = find_csys(GLOBAL_NAME)) == NULL) {
			p_error(
				"Cannot find default coordinate system for elt_geom_csys",
				NULL);
		}
	} else {
		if ((elt_geom_csys = find_csys(elt_geom_csys_name_E))
			== NULL) {
			p_error("Coord sys given for elt_geom_csys was never defined",
				NULL);
		}
	}

	/* Print titles
	--------------*/
	fprintf(ofp_E,"\n\n===================================================\n");
	fprintf(ofp_E,    "ELEMENT GEOMETRY (Organized by object)\n");
	fprintf(ofp_E,    "COORD SYS: %s\n",elt_geom_csys->name);
	fprintf(ofp_E,    "===================================================\n");

	/* Loop over elements
	--------------------*/
	current_obj = first_obj_E;
	current_elt = first_elt_E;
	while (current_elt != NULL) {

		fprintf(ofp_E,"\n");

		if (current_elt == current_obj->first_elt) {
			elt_num = 1;
			fprintf(ofp_E,"OBJECT: %s\n\n",current_obj->name);
			fprintf(ofp_E,ELT_GEOM_LABELS);
			fprintf(ofp_E,ELT_GEOM_UNDLNS);
		}

		for (i=0; i < current_elt->num_vertices; i++) {
			copy_vector(current_elt->disloc_seg[i].vert[0]->x,x);
			transform_position_vector(FORWARD_TRANS,
				elt_geom_csys->origin, elt_geom_csys->local_rot, x);
			if (i == 0) 
				copy_vector(x,first_vert_x);
			fprintf(ofp_E,ELT_GEOM_FMT,elt_num,
				current_elt->disloc_seg[i].vert[0]->name,x[0],x[1],x[2]);
		}
		fprintf(ofp_E,ELT_GEOM_FMT,elt_num,
			current_elt->disloc_seg[0].vert[0]->name,first_vert_x[0],
				first_vert_x[1],first_vert_x[2]);

		if (current_elt == current_obj->last_elt)
			current_obj = current_obj->next;

		elt_num++;
		current_elt = current_elt->next;
	}
}


/*********************** Function: array_max_norm ****************************
* Returns the maximum matrix norm (L-infinity norm) of the matrix a.
* Adapted from a similar function by Ken C. Cruikshank.
*
* In:	a			- the matrix for which the norm is to be found
*		start_row	- index of the matrix's start row
*		end_row		- index of the matrix's end row
*		start_col	- index of the matrix's start column
*		end_col		- index of the matrix's end column
******************************************************************************/
double   array_max_norm(double **a, int start_row, int end_row, int start_col,int end_col)
{
	int		i,j;
	double	norm;
	double	row_sum;

	norm = 0.0;
	for (i = start_row; i <= end_row; i++) {
		row_sum = 0.0;
		for (j = start_col; j <= end_col; j++) {
			row_sum += fabs(a[i][j]);
		}
		norm = MAX(norm,row_sum);
	}
	return(norm);
}


/*********************** Function: get_program_args *************************
* Prompts the user for input and output file names.  Only used if the
* symbolic constant FPROMPT is defined at compile time.
*****************************************************************************/
#ifdef FPROMPT
void     get_program_args(void)
{
	int		numwords;
	char	*word[MAXWORDS];
	char	line[MAXLINE];

	/* Get the input file name
	--------------------------*/
	printf("\n INPUT FILE: ");
	numwords = read_line(line,word);
	switch (numwords) {
		case 0:
			printf(  "             (Using default value)\n");
			break;
		case 1:
			if (strlen(word[0]) > MAXFILE-1) {
				p_error("File name too long",NULL);
			}
			strcpy(infile_E,word[0]);
			break;
		default:
			p_error("Invalid file name",NULL);
	}

	/* Get the output file name
	---------------------------*/
	printf("OUTPUT FILE: ");
	numwords = read_line(line,word);
	switch (numwords) {
		case 0:
			printf(  "             (Using default value)\n");
			break;
		case 1:
			if (strlen(word[0]) > MAXFILE-1) {
				p_error("File name too long",NULL);
			}
			strcpy(outfile_E,word[0]);
			break;
		default:
			p_error("Invalid file name",NULL);
	}
	printf("\n");
}
#endif
