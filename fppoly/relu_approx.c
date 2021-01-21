#include "relu_approx.h"

expr_t * create_relu_expr(neuron_t *out_neuron, neuron_t *in_neuron, size_t i, bool use_default_heuristics, bool is_lower){
	expr_t * res = alloc_expr();  
	res->inf_coeff = (double *)malloc(sizeof(double));
	res->sup_coeff = (double *)malloc(sizeof(double));
	res->inf_cst = 0.0;
	res->sup_cst = 0.0;
	res->type = SPARSE;
	res->size = 1;  
	res->dim = (size_t*)malloc(sizeof(size_t));
	res->dim[0] = i;
	double lb = in_neuron->lb;
	double ub = in_neuron->ub;
	double width = ub + lb;
	double lambda_inf = -ub/width;
	double lambda_sup = ub/width;
	
	if(ub<=0){
		res->inf_coeff[0] = 0.0;
		res->sup_coeff[0] = 0.0;
	}
	else if(lb<0){
		res->inf_coeff[0] = -1.0;
		res->sup_coeff[0] = 1.0;
	}
		
	else if(is_lower){
		double area1 = 0.5*ub*width;
		double area2 = 0.5*lb*width;
		if(use_default_heuristics){
			if(area1 < area2){
				res->inf_coeff[0] = 0.0;
				res->sup_coeff[0] = 0.0;
			}
			else{
				res->inf_coeff[0] = -1.0;
				res->sup_coeff[0] = 1.0;
			}
		}
		else{
				res->inf_coeff[0] = 0.0;
				res->sup_coeff[0] = 0.0;
		}
	}
	else{
		//Only upper bound contains bias cst
		double mu_inf = lambda_inf*lb;
		double mu_sup = lambda_sup*lb;
		res->inf_coeff[0] = lambda_inf;
		res->sup_coeff[0] = lambda_sup;
		res->inf_cst = mu_inf;
		res->sup_cst = mu_sup;
	}
	return res;
}


void handle_relu_layer(elina_manager_t *man, elina_abstract0_t* element, size_t num_neurons, size_t *predecessors, size_t num_predecessors, bool use_default_heuristics){
	
	assert(num_predecessors==1);
	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t numlayers = fp->numlayers;
	fppoly_add_new_layer(fp, num_neurons, predecessors, num_predecessors,true);
	neuron_t **out_neurons = fp->layers[numlayers]->neurons;
	// out_neurons, all the output neurons
	int k = predecessors[0]-1;
	neuron_t **in_neurons = fp->layers[k]->neurons;
	// Get the input neurons
	size_t i;
	for(i=0; i < num_neurons; i++){
		out_neurons[i]->twolb_flag = false;
		out_neurons[i]->lb = -fmax(0.0, -in_neurons[i]->lb);
		out_neurons[i]->ub = fmax(0,in_neurons[i]->ub);
		out_neurons[i]->lexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, use_default_heuristics, true);
		out_neurons[i]->uexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, use_default_heuristics, false);
		//No two lower bounds, so the auxilinary expression points to nothing
		out_neurons[i]-> aux_lexpr = NULL;
	}
}

expr_t * create_relu_two_lexpr(neuron_t *out_neuron, neuron_t *in_neuron, size_t i, bool use_default_heuristics, bool is_aux){
	expr_t * res = alloc_expr();  
	res->inf_coeff = (double *)malloc(sizeof(double));
	res->sup_coeff = (double *)malloc(sizeof(double));
	res->inf_cst = 0.0;
	res->sup_cst = 0.0;
	res->type = SPARSE;
	res->size = 1;  
	res->dim = (size_t*)malloc(sizeof(size_t));
	res->dim[0] = i;
	double lb = in_neuron->lb;
	double ub = in_neuron->ub;
	double width = ub + lb;
	double lambda_inf = -ub/width;
	double lambda_sup = ub/width;

	if(!is_aux){
		//If not auxilinary lower bound, we will store the one that deeppoly selected
		double area1 = 0.5*ub*width;
		double area2 = 0.5*lb*width;
		if(use_default_heuristics){
			if(area1 < area2){
				res->inf_coeff[0] = 0.0;
				res->sup_coeff[0] = 0.0;
			}
			else{
				res->inf_coeff[0] = -1.0;
				res->sup_coeff[0] = 1.0;
			}
		}
		else{
			res->inf_coeff[0] = 0.0;
			res->sup_coeff[0] = 0.0;
		}
	}
	else{
		//If it is auxilinary lower bound, we will store the one that deeppoly neglected
		double area1 = 0.5*ub*width;
		double area2 = 0.5*lb*width;
		if(use_default_heuristics){
			if(area1 < area2){
				res->inf_coeff[0] = -1.0;
				res->sup_coeff[0] = 1.0;
			}
			else{
				res->inf_coeff[0] = 0.0;
				res->sup_coeff[0] = 0.0;
			}
		}
		else{
			res->inf_coeff[0] = -1.0;
			res->sup_coeff[0] = 1.0;
		}
	}
	return res;
}

void handle_relu_layer_twolb(elina_manager_t *man, elina_abstract0_t* element, size_t num_neurons, size_t *predecessors, size_t num_predecessors, bool use_default_heuristics){
	
	assert(num_predecessors==1);
	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t numlayers = fp->numlayers;
	fppoly_add_new_layer(fp, num_neurons, predecessors, num_predecessors,true);
	neuron_t **out_neurons = fp->layers[numlayers]->neurons;
	// out_neurons, all the output neurons
	int k = predecessors[0]-1;
	neuron_t **in_neurons = fp->layers[k]->neurons;
	// Get the input neurons
	size_t i;
	
	for(i=0; i < num_neurons; i++){
		// For each neuron, handle the concrete and symbolic lower and upper bound
		out_neurons[i]->lb = -fmax(0.0, -in_neurons[i]->lb);
		out_neurons[i]->ub = fmax(0,in_neurons[i]->ub);
		if(out_neurons[i]->ub<=0){			
			out_neurons[i]->twolb_flag = false;
			out_neurons[i]->lexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, use_default_heuristics, true);
			out_neurons[i]->uexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, use_default_heuristics, false);
			out_neurons[i]->aux_lexpr = NULL;
		}
		else if(out_neurons[i]->lb<0){
			out_neurons[i]->twolb_flag = false;
			out_neurons[i]->lexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, use_default_heuristics, true);
			out_neurons[i]->uexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, use_default_heuristics, false);
			out_neurons[i]->aux_lexpr = NULL;
		}
		else{
			out_neurons[i]->twolb_flag = true;
			out_neurons[i]->lexpr = create_relu_two_lexpr(out_neurons[i], in_neurons[i], i, use_default_heuristics, false);
			out_neurons[i]->aux_lexpr = create_relu_two_lexpr(out_neurons[i], in_neurons[i], i, use_default_heuristics, true);
			out_neurons[i]->uexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, use_default_heuristics, false);

		}
		
	}
	
}
