#include "source.h"
using namespace dealii;




template<int dim>
class Elastic{
public:
  Elastic() : dof_handler(triangulation),
              fe(FE_Q<dim>(1), dim),
              max_cycle(5)
              {};
  void run();
private:
  double L2L2_error, L2H1_error;
  unsigned int max_cycle;
  void setup_system();
  void assemble_system();
  void solve();
  void output_result();
  void output_results(unsigned int);
  void compute_error();
  Triangulation<dim>   triangulation;
  DoFHandler<dim>      dof_handler;
  FESystem<dim>        fe;
  AffineConstraints<double> constraints;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs; 
  
  ConvergenceTable convergence_table;

}; //End of elastic class




template<int dim>
class BoundaryCondition : public Function<dim>{
public:
  BoundaryCondition() : Function<dim>(dim){}
  virtual double value(const Point<dim> &, 
                       const unsigned int component = 0) const override;

  virtual void vector_value(const Point<dim>&,
                             Vector<double> &) const override;
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p, const unsigned int component = 0) const override;

};// End of boundary condition class


template<int dim>
double BoundaryCondition<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const
{
  const double Pi = numbers::PI;
  const double arg = 2.0  * Pi * p[component];
  double return_value = std::sin(arg); 
  return return_value;
}

template<int dim>
void BoundaryCondition<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> & values) const
{
  for(unsigned int c = 0; c < this -> n_components; ++c)
    values(c) = BoundaryCondition<dim>::value(p,c);

}

template<int dim>
Tensor<1,dim> BoundaryCondition<dim>::gradient(const Point<dim> &p, 
                                 const unsigned int component) const
{
  double pi = numbers::PI;
  double scalar_value = 2.0 * pi * std::cos(2.0 * pi * p[component]);
  Tensor<1,dim> return_value;
  if(component == 0)
  {
    return_value[0] = scalar_value;
    return_value[1] = 0.0;
  }
  else
  {
    return_value[1] = scalar_value;
    return_value[0] = 0.0;
  } 
  return return_value; 
}


template<int dim>
class RHS : public Function<dim>{
public:
  RHS() : Function<dim>(dim){}
  virtual double value(const Point<dim> &, 
                       const unsigned int component = 0) const override;

  virtual void vector_value(const Point<dim>&,
                             Vector<double> &) const override;
};// End of boundary condition class


template<int dim>
double RHS<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const
{
  const double Pi = numbers::PI;
  const double arg = 2.0 * Pi * p[component];
  double return_value = 12.0 * std::pow(Pi,2) * std::sin(arg); 
  return return_value;
}

template<int dim>
void RHS<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> & values) const
{
  for(unsigned int c = 0; c < this -> n_components; ++c)
    values(c) = value(p,c);

}



template<int dim>
void Elastic<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}





template<int dim>
void Elastic<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> lambda_values(n_q_points);
  std::vector<double> mu_values(n_q_points);
  Functions::ConstantFunction<dim> lambda(1.), mu(1.);
  Vector<double> rhs_values(dim);
  RHS<dim> rhs;
  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    cell_matrix = 0;
    cell_rhs    = 0;
    fe_values.reinit(cell);
    lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
    mu.value_list(fe_values.get_quadrature_points(), mu_values);
    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      const Point<dim> xq = fe_values.quadrature_point(q_point);
      rhs.vector_value(xq, rhs_values); 
     for(unsigned int i = 0; i < dofs_per_cell; ++i)
     {
        const unsigned int component_i =
                      fe.system_to_component_index(i).first;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
           const unsigned int component_j =
                               fe.system_to_component_index(j).first;

          cell_matrix(i, j) +=
                        (                                                  
                        (fe_values.shape_grad(i, q_point)[component_i] * 
                         fe_values.shape_grad(j, q_point)[component_j] * 
                          lambda_values[q_point])                          
                        +                                                
                        (fe_values.shape_grad(i, q_point)[component_j] * 
                         fe_values.shape_grad(j, q_point)[component_i] * 
                          mu_values[q_point])                             
                        +                                                
                        ((component_i == component_j) ?        
                           (fe_values.shape_grad(i, q_point) * 
                            fe_values.shape_grad(j, q_point) * 
                            mu_values[q_point]  ) : 0
                        )                                  
                        ) *                                    
                        fe_values.JxW(q_point);
          } //end of j_point loop
          cell_rhs(i) += fe_values.shape_value(i, q_point) *
                         rhs_values[component_i] *
                         fe_values.JxW(q_point);
       } // end of i loop
     } //end of q loop

     cell->get_dof_indices(local_dof_indices);
     for(unsigned int i = 0; i < dofs_per_cell; ++i)
     {
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i,j));
        }//end of j loop
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
     }//end of i loop
   }//end of cell loop

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           BoundaryCondition<dim>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs); 


}



template<int dim>
void Elastic<dim>::solve()
{
  SolverControl solver_control(2000, 1e-12);
  SolverCG<>    cg(solver_control);
  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  
  cg.solve(system_matrix, solution, system_rhs, preconditioner);
}
template<int dim>
void Elastic<dim>::compute_error()
{
  Vector<float> difference_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    BoundaryCondition<dim>(),
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::L2_norm);

  const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);


  VectorTools::integrate_difference(dof_handler,
            solution,
            BoundaryCondition<dim>(),
            difference_per_cell,
            QGauss<dim>(fe.degree + 1),
            VectorTools::H1_seminorm);
  const double H1_error =
    VectorTools::compute_global_error(triangulation,
              difference_per_cell,
              VectorTools::H1_seminorm);

  
  L2L2_error = L2_error;
  L2H1_error = H1_error;
  

}

template <int dim>
void Elastic<dim>::output_results(unsigned int cycle)
{
 /* DataOut<dim> data_out;
  std::string SOLUTION = dim == 2 ? "solution-2d" : "solution-3d";
  SOLUTION += "-cycle-" + std::to_string(cycle) + ".vtk";
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output(SOLUTION);
  data_out.write_vtk(output);
*/
  static double Prev_L2L2 = 1;
  static double Prev_L2H1 = 1;
  static double Prev_hmax = 1;
  double L2L2_rate = 0;
  double L2H1_rate = 0;
  const unsigned int n_active_cells = triangulation.n_active_cells();
  const unsigned int n_dofs         = dof_handler.n_dofs();
  const double       max_cell_dia   = GridTools::maximal_cell_diameter(
                                                 triangulation);

  if(cycle != 0)
  {
    double L2L2_rate_top = std::log2(L2L2_error / Prev_L2L2);
    double L2H1_rate_top = std::log2(L2H1_error / Prev_L2H1);
    double rate_bottom = std::log2(max_cell_dia / Prev_hmax);
    L2L2_rate = L2L2_rate_top / rate_bottom;
    L2H1_rate = L2H1_rate_top / rate_bottom;
  }
    
  convergence_table.add_value("cycle",cycle);
  convergence_table.add_value("dofs",n_dofs);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("hmax", max_cell_dia);
  convergence_table.add_value("L2", L2L2_error);
  convergence_table.add_value("L2 Rate", L2L2_rate);
  convergence_table.add_value("H1", L2H1_error);
  convergence_table.add_value("H1 Rate", L2H1_rate);
  
  Prev_L2H1 = L2H1_error;  
  Prev_L2L2 = L2L2_error;
  Prev_hmax = max_cell_dia;
 
}


template<int dim>
void Elastic<dim>::run()
{
   GridGenerator::hyper_cube(triangulation, 0, 1);
   triangulation.refine_global(2);
   for(unsigned int cycle = 0; cycle < max_cycle; ++cycle)
   {
     std::cout << "Cycle: " << cycle << std::endl;
     std::cout << "   Number of active cells:       "
               << triangulation.n_active_cells() << std::endl;
     setup_system();
     std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
               << std::endl;
     assemble_system();
     solve();


     compute_error();
     output_results(cycle);
     triangulation.refine_global(1);
  }
 
  convergence_table.set_precision("L2",3);
  convergence_table.set_precision("H1",3);
  convergence_table.set_precision("L2 Rate",3);
  convergence_table.set_precision("H1 Rate",3);
  convergence_table.set_scientific("L2",true);
  convergence_table.set_scientific("H1",true);

  convergence_table.set_tex_caption("cells", "\\# cells");
  convergence_table.set_tex_caption("dofs", "\\# dofs");
  convergence_table.set_tex_caption("L2", "$L^2-error$");
  convergence_table.set_tex_format("cells", "r");
  convergence_table.set_tex_format("dofs", "r");

  convergence_table.write_text(std::cout);
  std::string filename = "convergence.txt";
  std::ofstream table_file(filename);
  convergence_table.write_tex(table_file);
 

}

int main()
{
  Elastic<2> elastic;
  elastic.run();
  return 0;
}
