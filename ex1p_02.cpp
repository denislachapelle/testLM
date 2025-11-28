//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
/*
For square-disc.mesh
Attributes 1–4 = the outer square
1: bottom edge (y=0): vertices 0→1→…→6
2: right edge (x=1): vertices 6→…→12
3: top edge (y=1): vertices 12→…→18
4: left edge (x=0): vertices 18→…→23→0
Attributes 5–8 = the inner boundary (the “hole”), split into four arcs
5: inner arc 24→25→…→30
6: inner arc 31→…→36
7: inner arc 37→…→42
8: inner arc 43→…→47→24
*/
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/octahedron.mesh -o 1
//               mpirun -np 4 ex1p -m ../data/periodic-annulus-sector.msh
//               mpirun -np 4 ex1p -m ../data/periodic-torus-sector.msh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               mpirun -np 4 ex1p -pa -d cuda
//               mpirun -np 4 ex1p -fa -d cuda
//               mpirun -np 4 ex1p -pa -d occa-cuda
//               mpirun -np 4 ex1p -pa -d raja-omp
//               mpirun -np 4 ex1p -pa -d ceed-cpu
//               mpirun -np 4 ex1p -pa -d ceed-cpu -o 4 -a
//               mpirun -np 4 ex1p -pa -d ceed-cpu -m ../data/square-mixed.mesh
//               mpirun -np 4 ex1p -pa -d ceed-cpu -m ../data/fichera-mixed.mesh
//             * mpirun -np 4 ex1p -pa -d ceed-cuda
//             * mpirun -np 4 ex1p -pa -d ceed-hip
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/beam-tet.mesh -pa -d ceed-cpu
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Poisson problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Class to adapt parent space to submesh space and vice-versa. 
//  y_lambda = Bs * ( T_p2s * x_parent )   // via Mult
//  y_parent = T_s2p * ( Bs^T * y_lambda )  // via MultTranspose
class SubmeshOperator : public Operator
{
public:
  SubmeshOperator(HypreParMatrix &Bs_, //operator on submesh.
                    ParFiniteElementSpace &fes_parent,
                    ParFiniteElementSpace &fes_sub,
                    ParFiniteElementSpace &fes_lambda)
  : Operator(fes_lambda.GetTrueVSize(), fes_parent.GetTrueVSize()),
    Bs(Bs_),
    T_p2s(&fes_parent, &fes_sub), // TransferMap parent to submesh.
    T_s2p(&fes_sub, &fes_parent), // TransferMap submesh to parent.
    GF1p(&fes_parent),
    GF2s(&fes_sub),
    GF3l(&fes_lambda),
    GF4p(&fes_parent)
  { }

  virtual void Mult(const Vector &xp, Vector &yl) const override
  {
    GF1p = xp;
    GF2s = 0.0;
    GF3l = 0.0;
    T_p2s.Transfer(GF1p, GF2s);    // xs = T * xp
    Bs.Mult(GF2s, GF3l);           // ys = Bs * xs
    yl = GF3l;
  }

   virtual void MultTranspose(const Vector &yl, Vector &yp) const override
  {
    GF3l = yl;
    GF2s = 0.0;
    GF4p = 0.0;
    Bs.MultTranspose(GF3l, GF2s); // xs = Bs^T * ys
    T_s2p.Transfer(GF2s, GF4p);   // yp = T * xs
    yp = GF4p;
  }

private:
  HypreParMatrix &Bs;
  mutable ParTransferMap T_p2s;     // parent -> submesh
  mutable ParTransferMap T_s2p;     // submesh -> parent (acts like T^T)
  mutable ParGridFunction GF1p, GF2s, GF3l, GF4p;
  
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;

   const char *device_config = "cpu";
   bool visualization = true;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   //
   //create the submesh for the square boundary.
   //
   Array<int> *square_bdr_attrs = new Array<int>;
   square_bdr_attrs->Append(1); square_bdr_attrs->Append(2);square_bdr_attrs->Append(3);square_bdr_attrs->Append(4);
   ParSubMesh *squaremesh = new ParSubMesh(ParSubMesh::CreateFromBoundary(pmesh, *square_bdr_attrs));
   if (Mpi::Root()) squaremesh->PrintInfo();

   //
   //create the submesh for the round boundary.
   //
   Array<int> *round_bdr_attrs = new Array<int>;
   round_bdr_attrs->Append(5); round_bdr_attrs->Append(7);round_bdr_attrs->Append(3);round_bdr_attrs->Append(8);
   ParSubMesh *roundmesh = new ParSubMesh(ParSubMesh::CreateFromBoundary(pmesh, *round_bdr_attrs));
   if (Mpi::Root()) roundmesh->PrintInfo();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   MFEM_VERIFY(order > 0, "order must be larger than zero.")
   fec = new H1_FECollection(order, dim);
   delete_fec = true;
   
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   
   //The next two spaces are for application of boundary conditions.
   //space for lagrange multiplier 1 relate to square,
   H1_FECollection *LM1FEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *LM1FESpace = new ParFiniteElementSpace(squaremesh, LM1FEC);
   int LM1nbrDof = LM1FESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << LM1nbrDof << " LM1FESpace degree of freedom\n";

   //space for lagrange multiplier 2 relate to round,
   H1_FECollection *LM2FEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *LM2FESpace = new ParFiniteElementSpace(roundmesh, LM2FEC);
   int LM2nbrDof = LM2FESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << LM2nbrDof << " LM2FESpace degree of freedom\n";   


   //The next two spaces are for application of boundary conditions.
   //space for u over the submesh 1 relate to square,
   H1_FECollection *squareFEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *squareFESpace = new ParFiniteElementSpace(squaremesh, squareFEC);
   int squarenbrDof = squareFESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << squarenbrDof << " squareFESpace degree of freedom\n";

   //space for lagrange multiplier 2 relate to round,
   H1_FECollection *circFEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *circFESpace = new ParFiniteElementSpace(roundmesh, circFEC);
   int circnbrDof = circFESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << circnbrDof << " circFESpace degree of freedom\n";   



   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the
   //     Diffusion domain integrator.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the parallel bilinear form.
   a.Assemble();
   a.Finalize();
   HypreParMatrix *pa = a.ParallelAssemble();


//
// MBLF for square boundary and LM1.
//
   ParMixedBilinearForm *MBLF_LM1 = new ParMixedBilinearForm(squareFESpace /*trial*/, LM1FESpace/*test*/);
   MBLF_LM1->AddDomainIntegrator(new MassIntegrator(one));
   MBLF_LM1->Assemble();
   MBLF_LM1->Finalize();
   HypreParMatrix *pMBLF_LM1 = MBLF_LM1->ParallelAssemble();
   SubmeshOperator *MBLF_LM1_map = new SubmeshOperator(*pMBLF_LM1, fespace, *squareFESpace, *LM1FESpace);

//
// MBLF for circular boundary and LM2.
//
   ParMixedBilinearForm *MBLF_LM2 = new ParMixedBilinearForm(circFESpace /*trial*/, LM2FESpace/*test*/);
   MBLF_LM2->AddDomainIntegrator(new MassIntegrator(one));
   MBLF_LM2->Assemble();
   MBLF_LM2->Finalize();
   HypreParMatrix *pMBLF_LM2 = MBLF_LM2->ParallelAssemble();
   SubmeshOperator *MBLF_LM2_map = new SubmeshOperator(*pMBLF_LM2, fespace, *circFESpace, *LM2FESpace);


   //
   // Linearform for LM1.
   //
   ParLinearForm LF_LM1(LM1FESpace);
   ConstantCoefficient minusone(-1.0);
   LF_LM1.AddDomainIntegrator(new DomainLFIntegrator(minusone));
   LF_LM1.Assemble();


   //
   // Linearform for LM2.
   //
   ParLinearForm LF_LM2(LM2FESpace);
   LF_LM2.AddDomainIntegrator(new DomainLFIntegrator(one));
   LF_LM2.Assemble();

   


   //
   // create the block structure.
   //
   Array<int> BlockOffsets(4);
   BlockOffsets[0]=0;
   BlockOffsets[1]=pa->Height(); 
   BlockOffsets[2]=pMBLF_LM1->Height();
   BlockOffsets[3]=pMBLF_LM2->Height();
   BlockOffsets.PartialSum();
      
   {
      std::ofstream out("out/BlockOffsets.txt");
      BlockOffsets.Print(out, 10);
   }

   BlockOperator *BlockOp = new BlockOperator(BlockOffsets);

   TransposeOperator *MBLF_LM1_map_T = new TransposeOperator(MBLF_LM1_map);
   TransposeOperator *MBLF_LM2_map_T = new TransposeOperator(MBLF_LM2_map);
   

   BlockOp->SetBlock(0, 0, pa);
   BlockOp->SetBlock(0, 1, MBLF_LM1_map_T);
   BlockOp->SetBlock(0, 2, MBLF_LM2_map_T);
   BlockOp->SetBlock(1, 0, MBLF_LM1_map);
   BlockOp->SetBlock(2, 0, MBLF_LM2_map);
      
      if (Mpi::Root())  cout  << BlockOp->Height() << " BlockOp->Height()" << endl;
      if (Mpi::Root())  cout  << BlockOp->Width() << " BlockOp->Width()" << endl;

      // the solution vector.
      Vector x(BlockOp->Height());

   //
   // the rhs vector.
   //

   BlockVector B(BlockOffsets);
   B.GetBlock(0) = b;
   B.GetBlock(1) = LF_LM1;
   B.GetBlock(2) = LF_LM2;
      

//
// Build the preconditioner for ...
// 
//
//                                [ M  B1^T B2^T ]
//                                [ B1   0   0   ]
//                                [ B2   0   0   ]

   // --- Schur approximations on each LM submesh (disjoint) ---
ParBilinearForm Mlam1(LM1FESpace);
Mlam1.AddDomainIntegrator(new MassIntegrator);
Mlam1.Assemble(); 
Mlam1.Finalize();
HypreParMatrix *HPM_Mlam1 = Mlam1.ParallelAssemble(); 

ParBilinearForm Mlam2(LM2FESpace); 
Mlam2.AddDomainIntegrator(new MassIntegrator);
Mlam2.Assemble(); 
Mlam2.Finalize();
HypreParMatrix *HPM_Mlam2 = Mlam2.ParallelAssemble(); 

// Simple solvers for S11 and S22 (treat ≈ α_i * Mlam_i)
auto *S1 = new CGSolver(); S1->SetRelTol(1e-8); S1->SetMaxIter(50); S1->SetPrintLevel(0);
S1->SetOperator(*HPM_Mlam1);

auto *S2 = new CGSolver(); S2->SetRelTol(1e-8); S2->SetMaxIter(50); S2->SetPrintLevel(0);
S2->SetOperator(*HPM_Mlam2);

// If you need scaling (e.g., constraint coefficients), wrap S1/S2 in a ScaledOperator.

// --- Full block-diagonal preconditioner ---
BlockDiagonalPreconditioner P(BlockOffsets);
P.SetDiagonalBlock(0, pa);  // primal
P.SetDiagonalBlock(1, S1);  // λ1 Schur
P.SetDiagonalBlock(2, S2);  // λ2 Schur
        





      
   // 13. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use Jacobi smoothing, for now.

   

   
   
   if(1)
   {
      FGMRESSolver solver(MPI_COMM_WORLD);
      solver.SetAbsTol(0);
      solver.SetRelTol(1e-6);
      solver.SetMaxIter(50000);
      solver.SetPrintLevel(1);
      solver.SetKDim(50);
      solver.SetOperator(*BlockOp);
      solver.SetPreconditioner(P);
      solver.Mult(B, x);
      if (Mpi::Root())  cout  << solver.GetFinalRelNorm() << " GetFinalRelNorm" << endl;
   }





/*
   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }
   */

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
