#include "source.h"
using namespace dealii;


template<int dim>
class Wave{
public:
  Wave(); 
  void run();
private:
  double time, time_step,
         time_final;
  unsigned int max_cycle,
               timestep_number;
  void setup_system();
  void assemble_system_u();
  void assemble_system_v();
  void solve_u(); 
  void solve_v();
  void output_result() const;
  Triangulation<dim>   triangulation;
  DoFHandler<dim>      dof_handler;
  FESystem<dim>        fe;
  AffineConstraints<double> constraints;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix_u;
  SparseMatrix<double> system_matrix_v;
  Vector<double>       system_rhs_u;
  Vector<double>       system_rhs_v; 
  Vector<double>       solution_u;
  Vector<double>       solution_v;
  Vector<double>       old_solution_u;
  Vector<double>       old_solution_v;

}; //End of elastic class


template<int dim>
class BoundaryConditionU : public Function<dim>{
public:
  BoundaryConditionU() : Function<dim>(dim){}
  virtual double value(const Point<dim> &, 
                       const unsigned int component = 0) const override;

  virtual void vector_value(const Point<dim>&,
                             Vector<double> &) const override;
 // virtual Tensor<1, dim>
  //gradient(const Point<dim> &p, const unsigned int component = 0) const override;

};// End of boundary condition class


template<int dim>
double BoundaryConditionU<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const
{
  double return_value; 
  if(component == 0)
  {
    const double time = this->get_time();
    const double Pi = numbers::PI;
    if((time <= 0.5) && (p[0] == -1.0) && (-1. / 3. < p[1])&&
       ( p[1] < 1. / 3.)) 
    {  
      const double arg = 4.0 * time  * Pi;
      return_value = std::sin(arg);
    }
    else
      return_value = 0.;
  }
  else
    return_value = 0.;
  
  return return_value;
}

template<int dim>
void BoundaryConditionU<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> & values) const
{
  for(unsigned int c = 0; c < this -> n_components; ++c)
    values(c) = BoundaryConditionU<dim>::value(p,c);

}

template<int dim>
class BoundaryConditionV : public Function<dim>{
public:
  BoundaryConditionV() : Function<dim>(dim){}
  virtual double value(const Point<dim> &, 
                       const unsigned int component = 0) const override;

  virtual void vector_value(const Point<dim>&,
                             Vector<double> &) const override;
 // virtual Tensor<1, dim>
  //gradient(const Point<dim> &p, const unsigned int component = 0) const override;

};// End of boundary condition class


template<int dim>
double BoundaryConditionV<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const
{
  double return_value; 
  if(component == 0)
  {
    const double time = this->get_time();
    const double Pi = numbers::PI;
    if((time <= 0.5) && (p[0] == -1.0) && (-1. / 3. < p[1])&&
       ( p[1] < 1. / 3.)) 
    {  
      const double arg = 4.0 * time  * Pi;
      return_value = 4.0* Pi* std::cos(arg);
    }
    else
      return_value = 0.;
  }
  else
    return_value = 0.;
  
  return return_value;
}

template<int dim>
void BoundaryConditionV<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> & values) const
{
  for(unsigned int c = 0; c < this -> n_components; ++c)
    values(c) = BoundaryConditionV<dim>::value(p,c);

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
  return 0.; 
}

template<int dim>
void RHS<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> & values) const
{
  for(unsigned int c = 0; c < this -> n_components; ++c)
    values(c) = value(p,c);

}


template<int dim>
class InitialVelocityU : public Function<dim>{
public:
  InitialVelocityU() : Function<dim>(dim){}
  virtual double value(const Point<dim> &, 
                       const unsigned int component = 0) const override;

  virtual void vector_value(const Point<dim>&,
                             Vector<double> &) const override;
};// End of boundary condition class


template<int dim>
double InitialVelocityU<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const
{
  return 0.; 
}

template<int dim>
void InitialVelocityU<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> & values) const
{
  for(unsigned int c = 0; c < this -> n_components; ++c)
    values(c) = value(p,c);

}

template<int dim>
class InitialVelocityV : public Function<dim>{
public:
  InitialVelocityV() : Function<dim>(dim){}
  virtual double value(const Point<dim> &, 
                       const unsigned int component = 0) const override;

  virtual void vector_value(const Point<dim>&,
                             Vector<double> &) const override;
};// End of boundary condition class


template<int dim>
double InitialVelocityV<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const
{
  return 0.; 
}

template<int dim>
void InitialVelocityV<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> & values) const
{
  for(unsigned int c = 0; c < this -> n_components; ++c)
    values(c) = value(p,c);

}




template<int dim>
Wave<dim>::Wave() 
  : dof_handler(triangulation),
    fe(FE_Q<dim>(1), dim),
    max_cycle(1),
    time_final(5.0),
    time_step(1./100.),
    timestep_number(1)
  {};


template<int dim>
void Wave<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  solution_u.reinit(dof_handler.n_dofs());
  solution_v.reinit(dof_handler.n_dofs());
  old_solution_u.reinit(dof_handler.n_dofs());
  old_solution_v.reinit(dof_handler.n_dofs());
  system_rhs_u.reinit(dof_handler.n_dofs());
  system_rhs_v.reinit(dof_handler.n_dofs());
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix_u.reinit(sparsity_pattern);
  system_matrix_v.reinit(sparsity_pattern);
}

template<int dim>
void Wave<dim>::assemble_system_u()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();
  FullMatrix<double> cell_matrix_u(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs_u(dofs_per_cell);
  Vector<double>     rhs_values_now(dim);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> mu_values(n_q_points);
  std::vector<Vector<double>> old_values_u(n_q_points, Vector<double>(dim));
  std::vector<Vector<double>> old_values_v(n_q_points, Vector<double>(dim));
  Functions::ConstantFunction<dim> mu(1.);
  RHS<dim> rhs_now;
  rhs_now.set_time(time); 
  system_matrix_u = 0.0;
  system_rhs_u    = 0.0;
  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    cell_matrix_u = 0.0;
    cell_rhs_u    = 0.0;
    fe_values.reinit(cell);
    fe_values.get_function_values(old_solution_u,old_values_u);
    fe_values.get_function_values(old_solution_v,old_values_v);
    mu.value_list(fe_values.get_quadrature_points(), mu_values);
    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
     const Point<dim> xq = fe_values.quadrature_point(q_point);
     rhs_now.vector_value(xq,rhs_values_now);
     for(unsigned int i = 0; i < dofs_per_cell; ++i)
     {
        const unsigned int component_i =
                      fe.system_to_component_index(i).first;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
           const unsigned int component_j =
                               fe.system_to_component_index(j).first;

           cell_matrix_u(i,j) += (component_i == component_j) ? fe_values.shape_value(i,q_point) *
                                                                fe_values.shape_value(j,q_point) *
                                                                fe_values.JxW(q_point): 0;
           cell_matrix_u(i,j) += ((component_i == component_j) ?
                                 (fe_values.shape_grad(i, q_point) *
                                  fe_values.shape_grad(j, q_point) *
                                  mu_values[q_point]*
                                  fe_values.JxW(q_point)*
                                  time_step * time_step ) : 0
                                 );
          } //end of j_point loop
          cell_rhs_u(i) +=(
                          fe_values.shape_value(i, q_point) *
                          rhs_values_now[component_i] 
                         )*fe_values.JxW(q_point) * time_step * time_step; //(k_n)^2 * f(x_q,t_n)  
          cell_rhs_u(i) +=(
                         old_values_v[q_point][component_i]*time_step
                         +
                         old_values_u[q_point][component_i]
                        )*fe_values.shape_value(i, q_point)*fe_values.JxW(q_point);   // k_n * M * v^{n-1} + M * u^{n-1}
       } // end of i loop
     } //end of q loop

     cell->get_dof_indices(local_dof_indices);
     for(unsigned int i = 0; i < dofs_per_cell; ++i)
     {
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          system_matrix_u.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix_u(i,j));
        }//end of j loop
        system_rhs_u(local_dof_indices[i]) += cell_rhs_u(i);
     }//end of i loop
   }//end of cell loop

  BoundaryConditionU<dim> boundary_function;
  boundary_function.set_time(time);
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           boundary_function,
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix_u,
                                     solution_u,
                                     system_rhs_u);
}

template<int dim>
void Wave<dim>::assemble_system_v()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();
  FullMatrix<double> cell_matrix_v(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs_v(dofs_per_cell);
  Vector<double>     rhs_values_now(dim);
  std::vector<std::vector<Tensor<1,dim>>> u_grad(n_q_points,std::vector<Tensor<1,dim>>(dim));
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<Vector<double>> old_values_v(n_q_points, Vector<double>(dim));
  RHS<dim> rhs_now;
  rhs_now.set_time(time); 
  system_matrix_v = 0.0;
  system_rhs_v = 0.0;
  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    cell_matrix_v = 0.0;
    cell_rhs_v    = 0.0;
    fe_values.reinit(cell);
    fe_values.get_function_values(old_solution_v,old_values_v);
    fe_values.get_function_gradients(solution_u,u_grad);
    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
     const Point<dim> xq = fe_values.quadrature_point(q_point);
     rhs_now.vector_value(xq,rhs_values_now);
     for(unsigned int i = 0; i < dofs_per_cell; ++i)
     {
        const unsigned int component_i =
                      fe.system_to_component_index(i).first;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
           const unsigned int component_j =
                               fe.system_to_component_index(j).first;

           cell_matrix_v(i,j) += (component_i == component_j) ? fe_values.shape_value(i,q_point) *
                                                                fe_values.shape_value(j,q_point) *
                                                                fe_values.JxW(q_point): 0;
        } //end of j_point loop
          cell_rhs_v(i) +=(
                           old_values_v[q_point][component_i]*
                           fe_values.shape_value(i, q_point) // M * v^{n-1}
                          )*fe_values.JxW(q_point);
          cell_rhs_v(i) +=(
                           fe_values.shape_value(i, q_point) *
                           rhs_values_now[component_i] * time_step //(k_n)* f(x_q,t_n)  
                          )*fe_values.JxW(q_point);
          cell_rhs_v(i) -=(
                           u_grad[q_point][component_i] * 
                           fe_values.shape_grad(i, q_point)
                          )*fe_values.JxW(q_point) * time_step;
     } // end of i loop
    } //end of q loop
     cell->get_dof_indices(local_dof_indices);
     for(unsigned int i = 0; i < dofs_per_cell; ++i)
     {
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          system_matrix_v.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix_v(i,j));
        }//end of j loop
        system_rhs_v(local_dof_indices[i]) += cell_rhs_v(i);
     }//end of i loop
   }//end of cell loop
  BoundaryConditionV<dim> boundary_function;
  boundary_function.set_time(time);
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           boundary_function,
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix_v,
                                     solution_v,
                                     system_rhs_v);
}
template<int dim>
void Wave<dim>::output_result() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<std::string> solution_names;
  switch(dim)
  {
        case 1:
          solution_names.emplace_back("solution");
          break;
        case 2:
          solution_names.emplace_back("U1_solution");
          solution_names.emplace_back("U2_solution");
          break;
        case 3:
          solution_names.emplace_back("U1_solution");
          solution_names.emplace_back("U2_solution");
          solution_names.emplace_back("U3_solution");
          break;
        default:
          Assert(false, ExcNotImplemented());
   }
  data_out.add_data_vector(solution_u, solution_names);
  data_out.build_patches();
  std::ofstream output("solution-" + std::to_string(timestep_number)
                       + ".vtk");
  data_out.write_vtk(output);
}
template<int dim>
void Wave<dim>::solve_u()
{
  SolverControl solver_control(5000, 1e-12);
  SolverCG<>    cg(solver_control);
  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix_u, 1.2);
  cg.solve(system_matrix_u, solution_u, system_rhs_u, preconditioner);
}
template<int dim>
void Wave<dim>::solve_v()
{
  SolverControl solver_control(5000, 1e-12);
  SolverCG<>    cg(solver_control);
  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix_v, 1.2);
  cg.solve(system_matrix_v, solution_v, system_rhs_v, preconditioner);
}

template<int dim>
void Wave<dim>::run()
{
   GridGenerator::hyper_cube(triangulation, -1, 1);
   triangulation.refine_global(7);
   for(unsigned int cycle = 0; cycle < max_cycle; ++cycle)
   {
     std::cout << "Cycle: " << cycle << std::endl;
     std::cout << "   Number of active cells:       "
               << triangulation.n_active_cells() << std::endl;
     setup_system();
     std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
               << std::endl;
     time = 0.;
     InitialVelocityU<dim> initial_U;
     InitialVelocityV<dim> initial_V;
     initial_U.set_time(time);
     initial_V.set_time(time);
     VectorTools::interpolate(dof_handler,
                              initial_U,
                              old_solution_u);
     
     VectorTools::interpolate(dof_handler,
                              initial_V,
                              old_solution_v);
     time += time_step;
     while(0.0 <= time_final - time)
     {
       assemble_system_u();
       solve_u();
       assemble_system_v();
       solve_v();
       old_solution_u = solution_u;
       old_solution_v = solution_v;
       output_result();
       time += time_step;
       timestep_number++;
       std::cout << "Time = " << time 
       << "  Time Number = " << timestep_number
       << std::endl;
    }
  }
 
}

int main()
{
  Wave<2> wave;
  wave.run();
  return 0;
}
