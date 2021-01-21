#ifdef GUROBI
#include <stdlib.h>
#include <stdio.h>

#include "gurobi_c.h"
#endif

#include "compute_bounds.h"
#include "math.h"

#define MAXNUM_EXPR 2048

int num_expr_accu = 0;
int neu_count = 0;
expr_t * replace_input_poly_cons_in_lexpr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp){
	size_t dims = expr->size;
	size_t i,k;
	double tmp1, tmp2;
	expr_t * res;
	if(expr->type==DENSE){
		k = 0;
	}
	else{
		k = expr->dim[0];		
	}
	expr_t * mul_expr = NULL;
			
	if(expr->sup_coeff[0] <0){
		mul_expr = fp->input_uexpr[k];
	}
	else if(expr->inf_coeff[0] < 0){
		mul_expr = fp->input_lexpr[k];
	}
		
	if(mul_expr!=NULL){
		if(mul_expr->size==0){
			res = multiply_cst_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		else{
			res = multiply_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
	}
		
	else{
		elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,expr->inf_coeff[0],expr->sup_coeff[0],fp->input_inf[k],fp->input_sup[k]);
		res = create_cst_expr(tmp1, -tmp1);
	}
	for(i=1; i < dims; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
			
		expr_t * mul_expr = NULL;
		expr_t * sum_expr = NULL;
		if(expr->sup_coeff[i] <0){
			mul_expr = fp->input_uexpr[k];
		}
		else if(expr->inf_coeff[i] <0){
			mul_expr = fp->input_lexpr[k];
		}
			
		if(mul_expr!=NULL){
			if(mul_expr->size==0){
				sum_expr = multiply_cst_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_cst_expr(pr,res,sum_expr);
			}	
			else if(expr->inf_coeff[i]!=0 && expr->sup_coeff[i]!=0){
				sum_expr = multiply_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_expr(pr,res,sum_expr);
			}
				//free_expr(mul_expr);
			if(sum_expr!=NULL){
				free_expr(sum_expr);
			}
		}
		else{
			elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
			res->inf_cst = res->inf_cst + tmp1;
			res->sup_cst = res->sup_cst - tmp1;
		}
	}
		
	res->inf_cst = res->inf_cst + expr->inf_cst; 
	res->sup_cst = res->sup_cst + expr->sup_cst; 
	return res;
}


expr_t * replace_input_poly_cons_in_uexpr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp){
	size_t dims = expr->size;
	size_t i,k;
	double tmp1, tmp2;
	expr_t * res;
	if(expr->type==DENSE){
		k = 0;
	}
	else{
		k = expr->dim[0];		
	}
	expr_t * mul_expr = NULL;
			
	if(expr->sup_coeff[0] <0){
		mul_expr = fp->input_lexpr[k];
	}
	else if(expr->inf_coeff[0] < 0){
		mul_expr = fp->input_uexpr[k];
	}
		
	if(mul_expr!=NULL){
		if(mul_expr->size==0){
			res = multiply_cst_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		else{
			res = multiply_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
	}
	else{
		elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,expr->inf_coeff[0],expr->sup_coeff[0],fp->input_inf[k],fp->input_sup[k]);
		res = create_cst_expr(-tmp2, tmp2);
	}
                //printf("finish\n");
		//fflush(stdout);
	for(i=1; i < dims; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
		expr_t * mul_expr = NULL;
		expr_t * sum_expr = NULL;
		if(expr->sup_coeff[i] <0){
			mul_expr = fp->input_lexpr[k];
		}
		else if(expr->inf_coeff[i] <0){
			mul_expr = fp->input_uexpr[k];
		}
			
		if(mul_expr!=NULL){
			if(mul_expr->size==0){
				sum_expr = multiply_cst_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_cst_expr(pr,res,sum_expr);
			}	
			else if(expr->inf_coeff[i]!=0 && expr->sup_coeff[i]!=0){
				sum_expr = multiply_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_expr(pr,res,sum_expr);
			}
				//free_expr(mul_expr);
			if(sum_expr!=NULL){
				free_expr(sum_expr);
			}
		}
		else{
			elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
			res->inf_cst = res->inf_cst - tmp2;
			res->sup_cst = res->sup_cst + tmp2;
		}
	}
	res->inf_cst = res->inf_cst + expr->inf_cst; 
	res->sup_cst = res->sup_cst + expr->sup_cst; 
	return res;
}

#ifdef GUROBI
void handle_gurobi_error(int error, GRBenv *env) {
    if (error) {
        printf("Gurobi error: %s\n", GRBgeterrormsg(env));
        exit(1);
    }
}

double substitute_spatial_gurobi(expr_t *expr, fppoly_t *fp, const int opt_sense) {

    GRBenv *env = NULL;
    GRBmodel *model = NULL;

    int error;

    error = GRBemptyenv(&env);
    handle_gurobi_error(error, env);
    error = GRBsetintparam(env, "OutputFlag", 0);
    handle_gurobi_error(error, env);
    error = GRBsetintparam(env, "NumericFocus", 2);
    handle_gurobi_error(error, env);
    error = GRBstartenv(env);
    handle_gurobi_error(error, env);

    double *lb, *ub, *obj;
    const size_t dims = expr->size;
    const size_t numvars = 3 * dims;

    lb = malloc(numvars * sizeof(double));
    ub = malloc(numvars * sizeof(double));
    obj = malloc(numvars * sizeof(double));

    for (size_t i = 0; i < dims; ++i) {
        const size_t k = expr->type == DENSE ? i : expr->dim[i];
        lb[i] = -fp->input_inf[k];
        ub[i] = fp->input_sup[k];
        obj[i] = opt_sense == GRB_MINIMIZE ? -expr->inf_coeff[i] : expr->sup_coeff[i];

        for (size_t j = 0; j < 2; ++j) {
            const size_t l = fp->input_uexpr[k]->dim[j];
            lb[dims + 2 * i + j] = -fp->input_inf[l];
            ub[dims + 2 * i + j] = fp->input_sup[l];
            obj[dims + 2 * i + j] = 0;
        }
    }

    error = GRBnewmodel(env, &model, NULL, numvars, obj, lb, ub, NULL, NULL);
    handle_gurobi_error(error, env);
    error = GRBsetintattr(model, "ModelSense", opt_sense);
    handle_gurobi_error(error, env);

    for (size_t i = 0; i < dims; ++i) {
        const size_t k = expr->type == DENSE ? i : expr->dim[i];

        int ind[] = {i, dims + 2 * i, dims + 2 * i + 1};

        double lb_val[] = {
            -1, -fp->input_lexpr[k]->inf_coeff[0], -fp->input_lexpr[k]->inf_coeff[1]
        };
        error = GRBaddconstr(model, 3, ind, lb_val, GRB_LESS_EQUAL, fp->input_lexpr[k]->inf_cst, NULL);
        handle_gurobi_error(error, env);

        double ub_val[] = {
            1, -fp->input_uexpr[k]->sup_coeff[0], -fp->input_uexpr[k]->sup_coeff[1]
        };
        error = GRBaddconstr(model, 3, ind, ub_val, GRB_LESS_EQUAL, fp->input_uexpr[k]->sup_cst, NULL);
        handle_gurobi_error(error, env);
    }

    size_t idx, nbr, s_idx, s_nbr;
    const size_t num_pixels = fp->num_pixels;

    for (size_t i = 0; i < fp->spatial_size; ++i) {
        idx = fp->spatial_indices[i];
        nbr = fp->spatial_neighbors[i];

        if (expr->type == DENSE) {
            s_idx = idx;
            s_nbr = nbr;
        } else {
            s_idx = s_nbr = num_pixels;

            for (size_t j = 0; j < dims; ++j) {
                if (expr->dim[j] == idx) {
                    s_idx = j;
                }
                if (expr->dim[j] == nbr) {
                    s_nbr = j;
                }
            }

            if ((s_idx == num_pixels) || (s_nbr == num_pixels)) {
                continue;
            }
        }

        int ind_x[] = {dims + 2 * s_idx, dims + 2 * s_nbr};
        int ind_y[] = {dims + 2 * s_idx + 1, dims + 2 * s_nbr + 1};
        double val[] = {1., -1.};

        error = GRBaddconstr(model, 2, ind_x, val, GRB_LESS_EQUAL, fp->spatial_gamma, NULL);
        handle_gurobi_error(error, env);
        error = GRBaddconstr(model, 2, ind_y, val, GRB_LESS_EQUAL, fp->spatial_gamma, NULL);
        handle_gurobi_error(error, env);
        error = GRBaddconstr(model, 2, ind_x, val, GRB_GREATER_EQUAL, -fp->spatial_gamma, NULL);
        handle_gurobi_error(error, env);
        error = GRBaddconstr(model, 2, ind_y, val, GRB_GREATER_EQUAL, -fp->spatial_gamma, NULL);
        handle_gurobi_error(error, env);
    }

    int opt_status;
    double obj_val;

    error = GRBoptimize(model);
    handle_gurobi_error(error, env);
    error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &opt_status);
    handle_gurobi_error(error, env);
    error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &obj_val);
    handle_gurobi_error(error, env);

    if (opt_status != GRB_OPTIMAL) {
        printf("Gurobi model status not optimal %i\n", opt_status);
        exit(1);
    }

    free(lb);
    free(ub);
    free(obj);

    GRBfreemodel(model);
    GRBfreeenv(env);

    return obj_val;
}
#endif

double compute_lb_from_expr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp, int layerno){
//This function will directly use the concrete bounds of variables shown in the expression to compute the value
//Can handle both expression replaced in intermediate layer or the input layer.
#ifdef GUROBI
    if ((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1 && fp->spatial_size > 0) {
        return expr->inf_cst - substitute_spatial_gurobi(expr, fp, GRB_MINIMIZE);
    }
#endif

	size_t i,k;
	double tmp1, tmp2;
	if((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1){
		expr =  replace_input_poly_cons_in_lexpr(pr, expr, fp);
	}
	size_t dims = expr->size;
	double res_inf = expr->inf_cst;
	if(expr->inf_coeff==NULL || expr->sup_coeff==NULL){
		return res_inf;
	}
	for(i=0; i < dims; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
		if(layerno==-1){
			//Until the input layer, use the bounds for input neurons to do the ocmputation
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
		}
		else{
			//Compute using each of the variable, by passing the concrete bounds
			//printf("constant values are %f, %f\n", fp->layers[layerno]->neurons[k]->lb,fp->layers[layerno]->neurons[k]->ub);
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->layers[layerno]->neurons[k]->lb,fp->layers[layerno]->neurons[k]->ub);
		}
		//All the values + the bias in the expression, res_inf is the final value
		//printf("res_inf and temp1 before computation are %f, %f\n", res_inf, tmp1);
		res_inf = res_inf + tmp1;		
		//printf("res_inf after computation are %f\n", res_inf);
	}
    if(fp->input_lexpr!=NULL && fp->input_uexpr!=NULL && layerno==-1){
		free_expr(expr);
	}
 	//printf("The inf coeffs in the lexpr are %f and %f\n",expr->inf_coeff[0],expr->inf_coeff[1]);
        //printf("compute lb from expr returns %f\n",res_inf);
        //fflush(stdout);
	return res_inf;
}

double compute_ub_from_expr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp, int layerno){
#ifdef GUROBI
    if ((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1 && fp->spatial_size > 0) {
        return expr->sup_cst + substitute_spatial_gurobi(expr, fp, GRB_MAXIMIZE);
    }
#endif

	size_t i,k;
	double tmp1, tmp2;

	if((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1){
		expr =  replace_input_poly_cons_in_uexpr(pr, expr, fp);
	}

	size_t dims = expr->size;
	double res_sup = expr->sup_cst;
	if(expr->inf_coeff==NULL || expr->sup_coeff==NULL){
		return res_sup;
	}
	for(i=0; i < dims; i++){
		//if(expr->inf_coeff[i]<0){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}		
		if(layerno==-1){
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
		}
		else{
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->layers[layerno]->neurons[k]->lb,fp->layers[layerno]->neurons[k]->ub);
		}
		res_sup = res_sup + tmp2;
			
	}
	//printf("sup: %g\n",res_sup);
	//fflush(stdout);
	if(fp->input_lexpr!=NULL && fp->input_uexpr!=NULL && layerno==-1){
		free_expr(expr);
	}
	return res_sup;
}


double get_lb_using_predecessor_layer(fppoly_internal_t * pr,fppoly_t *fp, expr_t **lexpr_ptr, int k){
	expr_t * tmp_l;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *lexpr = *lexpr_ptr;
	double res = INFINITY;
	res = compute_lb_from_expr(pr,lexpr,fp,k);
	tmp_l = lexpr;
	*lexpr_ptr = lexpr_replace_bounds(pr,lexpr,aux_neurons, fp->layers[k]->is_activation);
	free_expr(tmp_l);
	return res;
}

//begin of my new functions
double get_lbs_using_predecessor_layer(fppoly_internal_t * pr,fppoly_t *fp, expr_t **lexpr_ptr, int k, expr_list_t * listofexprs, bool random_prune){
	//The invocation of this function already mean that two slbs are allowed
	expr_t * tmp_l;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *lexpr = *lexpr_ptr;
	double res = INFINITY;
	res = compute_lb_from_expr(pr,lexpr,fp,k);
	tmp_l = lexpr;
	*lexpr_ptr = mullexpr_replace_bounds(pr,lexpr,aux_neurons, fp->layers[k]->is_activation, listofexprs, random_prune);
	free_expr(tmp_l);
	return res;
}

double get_ubs_using_predecessor_layer(fppoly_internal_t * pr,fppoly_t *fp, expr_t **uexpr_ptr, int k, expr_list_t * listofexprs, bool random_prune){
	expr_t * tmp_u;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *uexpr = *uexpr_ptr;
	double res = INFINITY;
	tmp_u = uexpr;
	res = compute_ub_from_expr(pr,uexpr,fp,k);
	*uexpr_ptr = muluexpr_replace_bounds(pr,uexpr,aux_neurons, fp->layers[k]->is_activation, listofexprs, random_prune);
	free_expr(tmp_u);
	return res;
}

//end of my new functions

double get_ub_using_predecessor_layer(fppoly_internal_t * pr,fppoly_t *fp, expr_t **uexpr_ptr, int k){
	expr_t * tmp_u;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *uexpr = *uexpr_ptr;
	double res = INFINITY;
	tmp_u = uexpr;
	res = compute_ub_from_expr(pr,uexpr,fp,k);
	*uexpr_ptr = uexpr_replace_bounds(pr,uexpr,aux_neurons, fp->layers[k]->is_activation);
	free_expr(tmp_u);
	return res;
}

expr_list_t * create_exprlist(bool has_two_slbs){
	if (has_two_slbs){
		expr_list_t *list = (expr_list_t*)malloc(sizeof(expr_list_t));
		list->numexprs = 0;
		list->expr_list = (expr_t **)malloc(MAXNUM_EXPR*sizeof(expr_t*));
		return list;
	}
	else
		return NULL;
}

void merge_expression_in_list(expr_list_t * source){
	size_t list_len, count, i;
	//list_len = sink->numexprs;
	size_t num_neurons = source->expr_list[0]->size;
	for (count=2; count<source->numexprs; count++){
		expr_t * first = source->expr_list[1];
		expr_t * curr = source->expr_list[count];
		for (i = 0; i < num_neurons; i++){
			first->inf_coeff[i] += curr->inf_coeff[i];
			first->sup_coeff[i] += curr->sup_coeff[i];
		}
		first->inf_cst += curr->inf_cst;
		first->sup_cst += curr->sup_cst;
		free_expr(curr);
	}
	if (source->numexprs > 1){
		expr_t * first = source->expr_list[1];
		for (i = 0; i < num_neurons; i++){
			first->inf_coeff[i] = first->inf_coeff[i]/(source->numexprs-1);
			first->sup_coeff[i] = first->sup_coeff[i]/(source->numexprs-1);
		}
		first->inf_cst = first->inf_cst/(source->numexprs-1);
		first->sup_cst = first->sup_cst/(source->numexprs-1);	
		source->numexprs=2;
	}
}
void print_expr(expr_t * source){
	size_t num_neurons; 
	num_neurons = source->size;
	size_t count;
        printf("Print this expr: ");
	for (count=0; count< num_neurons; count++){
		printf("%.2f ",source->sup_coeff[count]);
	}
	printf(";;;; ");
	printf("Bias: %.2f    End of print this expr\n",source->sup_cst);
	fflush(stdout);
}
double get_lb_using_previous_layers(elina_manager_t *man, fppoly_t *fp, expr_t *expr, size_t layerno, bool has_two_slbs, bool random_prune){
	size_t i, count, counter;
	int k;
	expr_t * lexpr = copy_expr(expr);
	expr_list_t * exprlist = create_exprlist(has_two_slbs);
	if (has_two_slbs){
		exprlist->expr_list[0] = copy_expr(expr);
		exprlist->numexprs++;
	}
    fppoly_internal_t * pr = fppoly_init_from_manager(man,ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	if(fp->numlayers==layerno){
		//If this is the last layer of the network? Then the predecssor layer would be the second last layer
		k = layerno-1;
	}
	else if((fp->layers[layerno]->is_concat == true) || (fp->layers[layerno]->num_predecessors==2)){
		//For concatenation and residual layer)
		k = layerno;
	}
	else{
		//The normal case
		k = fp->layers[layerno]->predecessors[0]-1;
	}	
	double res = INFINITY;

	while(k >=0){
	        if(fp->layers[k]->is_concat==true){
	        	//This branch handle the case where the layer is concatenation_layer
				size_t i;
				size_t *C = fp->layers[k]->C;
				size_t *predecessors = fp->layers[k]->predecessors;
				size_t num_predecessors = fp->layers[k]->num_predecessors;
				int common_predecessor = INT_MAX;
				expr_t ** sub_expr = (expr_t**)malloc(num_predecessors*sizeof(expr_t*));
				for(i=0; i < num_predecessors; i++){
					int pred = predecessors[i]-1;
					if(pred < common_predecessor){
						common_predecessor = pred;
					}
					sub_expr[i] = extract_subexpr_concatenate(lexpr,i, C,fp->layers[k]->dims, fp->layers[k]->num_channels);
				}
				for(i=0; i < num_predecessors; i++){
					
					int iter = predecessors[i]-1;
					if(sub_expr[i]->size>0){	
						while(iter!=common_predecessor){
							get_lb_using_predecessor_layer(pr,fp, &sub_expr[i],  iter);
							iter = fp->layers[iter]->predecessors[0]-1;
						}
					}
				}
				double inf_cst = lexpr->inf_cst;
				double sup_cst = lexpr->sup_cst;
				free_expr(lexpr);
				bool flag = true;
				//lexpr = copy_expr(sub_expr[0]);
				for(i=0; i < num_predecessors; i++){
					if(sub_expr[i]->size>0){
						if(flag==true){
							lexpr = copy_expr(sub_expr[i]);
							flag = false;
						}
						else{
							add_expr(pr, lexpr, sub_expr[i]);
						}
						
					}
					free_expr(sub_expr[i]);
				}	
				lexpr->inf_cst = lexpr->inf_cst + inf_cst;
				lexpr->sup_cst = lexpr->sup_cst + sup_cst;
				free(sub_expr);
				k = common_predecessor;
			}
			else if(fp->layers[k]->num_predecessors==2){
			    //This branch basically handle residual layer
				expr_t * lexpr_copy = copy_expr(lexpr);
				lexpr_copy->inf_cst = 0;
				lexpr_copy->sup_cst = 0;
				size_t predecessor1 = fp->layers[k]->predecessors[0]-1;
				size_t predecessor2 = fp->layers[k]->predecessors[1]-1;
				
				char * predecessor_map = (char *)calloc(k,sizeof(char));
				// Assume no nested residual layers
				int iter = fp->layers[predecessor1]->predecessors[0]-1;
				while(iter>=0){
					predecessor_map[iter] = 1;
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  fp->layers[predecessor2]->predecessors[0]-1;
				int common_predecessor = 0;
				while(iter>=0){
					if(predecessor_map[iter] == 1){
						common_predecessor = iter;
						break;
					}
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				
				iter = predecessor1;
				while(iter!=common_predecessor){
					get_lb_using_predecessor_layer(pr,fp, &lexpr,  iter);
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  predecessor2;
				while(iter!=common_predecessor){
					get_lb_using_predecessor_layer(pr,fp, &lexpr_copy,  iter);
					iter = fp->layers[iter]->predecessors[0]-1;					
				}
				free(predecessor_map);
				add_expr(pr,lexpr,lexpr_copy);
				
				free_expr(lexpr_copy);
				
				// Assume at least one non-residual layer between two residual layers
				k = common_predecessor;		
				continue;
			}
			else {
				 if (!has_two_slbs){	
                	double pre_res = get_lb_using_predecessor_layer(pr,fp, &lexpr, k);
				 	res =fmin(res,pre_res);
			 	 }
			 	 else{
			 	 	//If it is the case for 2slbs
			 	 	size_t cur_len = exprlist->numexprs;
					double pre_res;
		 	 		for(count = 0; count < cur_len; count++){
						pre_res = get_lbs_using_predecessor_layer(pr,fp, &exprlist->expr_list[count], k, exprlist, random_prune);
		 	 			res =fmin(res,pre_res);
					}
					if(exprlist->numexprs > 2 && random_prune){
						merge_expression_in_list(exprlist);
					}
			 	 }
				 //Update the k to be predecessor of this predecessor, continue until replace to the input layer
				 k = fp->layers[k]->predecessors[0]-1;	
			}	
	}	
	if (!has_two_slbs){	
		double pre_res = compute_lb_from_expr(pr,lexpr,fp,-1);
		res = fmin(res,pre_res);
   	 }
   	 else{
        	num_expr_accu += exprlist->numexprs;
       		neu_count++;
			double pre_res;
    		for(counter = 0; counter < exprlist->numexprs; counter++){
 			print_expr(exprlist->expr_list[counter]);
			pre_res = compute_lb_from_expr(pr,exprlist->expr_list[counter], fp, -1);
			res =fmin(res,pre_res);
		}
		free_expr_list(exprlist);
    }
	free_expr(lexpr);
	return res;	
}   


double get_ub_using_previous_layers(elina_manager_t *man, fppoly_t *fp, expr_t *expr, size_t layerno, bool has_two_slbs, bool random_prune){
	size_t i, count, counter;
	int k;
	//size_t numlayers = fp->numlayers;
	expr_t * uexpr = copy_expr(expr);
	//pointer to list of expression, and initialize the first element to be the same as expr
	expr_list_t * exprlist = create_exprlist(has_two_slbs);
	if (has_two_slbs){
		exprlist->expr_list[0] = copy_expr(expr);
		exprlist->numexprs++;
	}
    fppoly_internal_t * pr = fppoly_init_from_manager(man,ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	if(fp->numlayers==layerno){
		k = layerno-1;
	}
	else if((fp->layers[layerno]->is_concat == true) || (fp->layers[layerno]->num_predecessors==2)){
		k = layerno;
	}
	else{
		k = fp->layers[layerno]->predecessors[0]-1;
	}	
	double res =INFINITY;
	while(k >=0){
		if(fp->layers[k]->is_concat==true){
            //sort_expr(lexpr);
            size_t i;
            size_t *C = fp->layers[k]->C;
            size_t *predecessors = fp->layers[k]->predecessors;
            size_t num_predecessors = fp->layers[k]->num_predecessors;
            int common_predecessor = INT_MAX;
            expr_t ** sub_expr = (expr_t**)malloc(num_predecessors*sizeof(expr_t*));
            //size_t index_start = 0;
            for(i=0; i < num_predecessors; i++){
                    int pred = predecessors[i]-1;
                    //size_t num_neurons = fp->layers[pred]->dims;
                    if(pred < common_predecessor){
                            common_predecessor = pred;
                    }
                    sub_expr[i] = extract_subexpr_concatenate(uexpr,i, C,fp->layers[k]->dims, fp->layers[k]->num_channels);
                    //index_start = index_start + num_neurons;
            }
            for(i=0; i < num_predecessors; i++){
                                int iter = predecessors[i]-1;
				if(sub_expr[i]->size>0){
                	while(iter!=common_predecessor){
                        	get_ub_using_predecessor_layer(pr,fp, &sub_expr[i],  iter);
                        	iter = fp->layers[iter]->predecessors[0]-1;
                	}
				}
            }
			double inf_cst = uexpr->inf_cst;
			double sup_cst = uexpr->sup_cst;
            free_expr(uexpr);
			bool flag = true;
            for(i=0; i < num_predecessors; i++){
				if(sub_expr[i]->size>0){
					if(flag==true){
						uexpr = copy_expr(sub_expr[i]);
						flag = false;
					}
					else{
                		add_expr(pr, uexpr, sub_expr[i]);
					}
				}
                free_expr(sub_expr[i]);
            }	
            free(sub_expr);
			uexpr->inf_cst = uexpr->inf_cst + inf_cst;
			uexpr->sup_cst = uexpr->sup_cst + sup_cst;
            k = common_predecessor;
        }	
		else if(fp->layers[k]->num_predecessors==2){
				expr_t * uexpr_copy = copy_expr(uexpr);
				uexpr_copy->inf_cst = 0;
				uexpr_copy->sup_cst = 0;
				size_t predecessor1 = fp->layers[k]->predecessors[0]-1;
				size_t predecessor2 = fp->layers[k]->predecessors[1]-1;
				
				char * predecessor_map = (char *)calloc(k,sizeof(char));
				// Assume no nested residual layers
				int iter = fp->layers[predecessor1]->predecessors[0]-1;
				while(iter>=0){
					predecessor_map[iter] = 1;
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  fp->layers[predecessor2]->predecessors[0]-1;
				int common_predecessor = 0;
				while(iter>=0){
					if(predecessor_map[iter] == 1){
						common_predecessor = iter;
						break;
					}
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				
				iter = predecessor1;
				while(iter!=common_predecessor){
					get_ub_using_predecessor_layer(pr,fp, &uexpr,  iter);
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  predecessor2;
				while(iter!=common_predecessor){
					get_ub_using_predecessor_layer(pr,fp, &uexpr_copy,  iter);
					iter = fp->layers[iter]->predecessors[0]-1;					
				}
				free(predecessor_map);
				add_expr(pr,uexpr,uexpr_copy);
				
				free_expr(uexpr_copy);
				
				// Assume at least one non-residual layer between two residual layers
				k = common_predecessor;
				
				continue;
			}
			else {
 				 if (!has_two_slbs){	
				 	res= fmin(res,get_ub_using_predecessor_layer(pr,fp, &uexpr, k));
			 	 }
			 	 else{
			 	 	//If it is the case for 2slbs
			 	 	size_t cur_len = exprlist->numexprs;
		 	 		for(count = 0; count < cur_len; count++){
		 	 			res =fmin(res,get_ubs_using_predecessor_layer(pr,fp, &exprlist->expr_list[count], k, exprlist, random_prune));
					}
					if(exprlist->numexprs > 2 && random_prune){
						merge_expression_in_list(exprlist);
					}
			 	 }
				 k = fp->layers[k]->predecessors[0]-1;
			}
			
	}
	if (!has_two_slbs){	
		res = fmin(res,compute_ub_from_expr(pr,uexpr,fp,-1)); 
    }
    else{
	num_expr_accu += exprlist->numexprs;
	neu_count++;
	for(counter = 0; counter < exprlist->numexprs; counter++){
			res =fmin(res,compute_ub_from_expr(pr,exprlist->expr_list[counter], fp, -1));
	}
		free_expr_list(exprlist);
    }
	free_expr(uexpr);
	return res;
}
