//                       MFEM Example 1 - Serial Version modified to apply
//                            Dirichelet condition using Lagrange multipliers. 
//
// Written by denis lachapelle, November 2025.
//
// Compile with: make ex1_05
//
// Sample runs:  ex1_05
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
//               
/* Description:
MFEM Example 1 - Serial Version modified to apply Dirichelet condition
using Lagrange multipliers.

In this version the preconditioner is not implemented.

Options are:
   fo: field order.
   lmo: lagrange multiplier order.
   pt: preconditioner type: 0-no preconditioner, 1-DSmoother block diagonal, 2-schur complement block diagonal.
*/

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

//*************************************************
// Class to adapt parent space to submesh space and vice-versa. 
//  y_lambda = Bs * ( T_p2s * x_parent )   // via Mult
//  y_parent = T_s2p * ( Bs^T * y_lambda )  // via MultTranspose
class SubmeshOperator : public Operator
{
public:
  SubmeshOperator(SparseMatrix &Bs_, //operator on submesh.
                    FiniteElementSpace &fes_parent,
                    FiniteElementSpace &fes_sub,
                    FiniteElementSpace &fes_lambda)
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
  SparseMatrix &Bs;
  mutable TransferMap T_p2s;     // parent -> submesh
  mutable TransferMap T_s2p;     // submesh -> parent (acts like T^T)
  mutable GridFunction GF1p, GF2s, GF3l, GF4p;
  
};
/* This class implement an operator applying the inverse of Schur complement.
   Ainv_: Operator applying the aproximative inverse of the system matrix.
   B_: Operator applying Lagrange multiplier matrix.
 */   
class InvSchurComp : public Solver   
{
    private:
       Solver &Ainv;
       SubmeshOperator &B;
       TransposeOperator *BT;
       GMRESSolver *solver;
       ProductOperator *AM1BT, *BAM1BT;
       public:
       InvSchurComp(Solver &Ainv_, SubmeshOperator &B_)
       : Solver(B_.Height()), Ainv(Ainv_), B(B_)
       {

         BT = new TransposeOperator(B);
         AM1BT = new ProductOperator(&Ainv, BT, false, false);
         BAM1BT = new ProductOperator(&B, AM1BT, false, false);

         solver = new GMRESSolver();
         solver->SetAbsTol(0);
         solver->SetRelTol(1e-2);
         solver->SetMaxIter(200);
         solver->SetPrintLevel(0);
         solver->SetKDim(20);
         solver->SetOperator(*BAM1BT);

       }

         virtual void Mult(const Vector &b, Vector &x) const override
         {
            x = 0.0;
            solver->Mult(b, x);
         }

         virtual void SetOperator(const Operator &op)
         {

         }
};

class CompleteSchurPreconditioner : public Solver
{
      private:
         InvSchurComp *SC1, *SC2;
         Solver &Ainv;
         SubmeshOperator &B1, &B2;
         int Size;
         const Array<int> &block_offsets;
      public:
         CompleteSchurPreconditioner(const Array<int> &block_offsets_, Solver &Ainv_, SubmeshOperator &B1_, SubmeshOperator &B2_)
         : Solver(Ainv_.Height() + B1_.Height() + B2_.Height()), Ainv(Ainv_), B1(B1_), B2(B2_), block_offsets(block_offsets_)
         {
            SC1 = new InvSchurComp(Ainv, B1);
            SC2 = new InvSchurComp(Ainv, B2);
            Size = Ainv_.Height() + B1_.Height() + B2_.Height();
         }

         virtual void Mult(const Vector &x, Vector &y) const override
         {
            MFEM_ASSERT(x.Size() == Size(), "Preconditioner input size mismatch.");
            MFEM_ASSERT(y.Size() == Size(), "Preconditioner output size mismatch.");

            // remove const only locally, we won't modify x through xx
            Vector &xx = const_cast<Vector &>(x);
            Vector x0(xx, 0, block_offsets[1]); // primal
            Vector x1(xx, block_offsets[1], block_offsets[2] - block_offsets[1]); // LM1
            Vector x2(xx, block_offsets[2], block_offsets[3] - block_offsets[2]); // LM2

            Vector y0(y, 0, block_offsets[1]); // primal
            Vector y1(y, block_offsets[1], block_offsets[2] - block_offsets[1]); // LM1
            Vector y2(y, block_offsets[2], block_offsets[3] - block_offsets[2]); // LM2

            Ainv.Mult(x0, y0);
            SC1->Mult(x1, y1);
            SC2->Mult(x2, y2);
         }

         virtual void SetOperator(const Operator &op)
         {

         }


};



/// preconditioner for
/// [ A   B1^T  B2^T ]
/// [ B1   0     0   ]
/// [ B2   0     0   ]
///
/// P^{-1} ≈ diag(A^{-1}, S1^{-1}, S2^{-1}),
/// using DSmoother.
class ThreeBlockDiagonalPreconditioner : public Solver
{
private:
   Array<int> block_offsets;
   const SparseMatrix &A, &M1, &M2;
   Solver *precA, *precM1, *precM2;
public:
   ThreeBlockDiagonalPreconditioner(const Array<int> &offsets,
                                    const SparseMatrix &A_,
                                    const SparseMatrix &M1_,
                                    const SparseMatrix &M2_)
      : Solver(offsets.Last()),
        block_offsets(offsets),
        A(A_), M1(M1_), M2(M2_)
   {
      height = width = block_offsets.Last();

      precA = new DSmoother(1, 1.0, 100);
      precM1 = new DSmoother(1, 1.0, 100);
      precM2 = new DSmoother(1, 1.0, 100);

      MFEM_VERIFY(A.Finalized(), "Matrix A must be finalized.");
      MFEM_VERIFY(M1.Finalized(), "Matrix M1 must be finalized.");
      MFEM_VERIFY(M2.Finalized(), "Matrix M2 must be finalized.");

      // --- Build diagonals and DSmoothers (inside the class) ---
         precA->SetOperator(A);
         precM1->SetOperator(M1);
         precM2->SetOperator(M2);
   }

   virtual ~ThreeBlockDiagonalPreconditioner() {}

   virtual void SetOperator(const Operator &op) override
   {
      MFEM_ASSERT(op.Height() == op.Width(), "Operator must be square.");
      MFEM_ASSERT(op.Height() == height, "Operator size must match block offsets.");
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_ASSERT(x.Size() == Size(), "Preconditioner input size mismatch.");
      MFEM_ASSERT(y.Size() == Size(), "Preconditioner output size mismatch.");

      // remove const only locally, we won't modify x through xx
      Vector &xx = const_cast<Vector &>(x);
      Vector x0(xx, 0, block_offsets[1]); // primal
      Vector x1(xx, block_offsets[1], block_offsets[2] - block_offsets[1]); // LM1
      Vector x2(xx, block_offsets[2], block_offsets[3] - block_offsets[2]); // LM2

      Vector y0(y, 0, block_offsets[1]); // primal
      Vector y1(y, block_offsets[1], block_offsets[2] - block_offsets[1]); // LM1
      Vector y2(y, block_offsets[2], block_offsets[3] - block_offsets[2]); // LM2

      precA->Mult(x0, y0);
      precM1->Mult(x1, y1);
      precM2->Mult(x2, y2);
   }
};



int main(int argc, char *argv[])
{

   // 2. Parse command-line options.
   int fieldOrder = 1;
   int lmOrder = 1;
   int precType = 0;

   const char *device_config = "cpu";
   bool visualization = true;
   
   OptionsParser args(argc, argv);
   args.AddOption(&fieldOrder, "-fo", "--fieldorder",
                  "Field Finite element order.");
   args.AddOption(&lmOrder, "-lmo", "--lmorder",
                  "Lagrange Multiplier Finite element order.");
   args.AddOption(&precType, "-pt", "--prectype",
                  "0: no preconditioner, 1: DSmoother Block Diag, 2: Schur Block Diag.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
      args.PrintOptions(cout);

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   const char *mesh_file = "../data/square-disc.mesh";
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }
   
   //
   //create the submesh for the square boundary.FEM_VERIFY(A.Finalized(), "Matrix A must be finalized.");
   //
   Array<int> *square_bdr_attrs = new Array<int>;
   square_bdr_attrs->Append(1); square_bdr_attrs->Append(2);square_bdr_attrs->Append(3);square_bdr_attrs->Append(4);
   SubMesh *squaremesh = new SubMesh(SubMesh::CreateFromBoundary(mesh, *square_bdr_attrs));
   squaremesh->PrintInfo();

   //
   //create the submesh for the round boundary.
   //
   Array<int> *round_bdr_attrs = new Array<int>;
   round_bdr_attrs->Append(5); round_bdr_attrs->Append(6);round_bdr_attrs->Append(7);round_bdr_attrs->Append(8);
   SubMesh *roundmesh = new SubMesh(SubMesh::CreateFromBoundary(mesh, *round_bdr_attrs));
   roundmesh->PrintInfo();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   MFEM_VERIFY(fieldOrder > 0, "Field order must be larger than zero.");
   MFEM_VERIFY(lmOrder > 0, "Lagrange Multiplier order must be larger than zero.");
   
   
   FiniteElementCollection *fec;
   bool delete_fec;
   fec = new H1_FECollection(fieldOrder, dim);
   delete_fec = true;
   
   FiniteElementSpace fespace(&mesh, fec);
   int size = fespace.GetTrueVSize();
   cout << "Number of finite element unknowns: " << size << endl;
      
   //The next two spaces are for application of boundary conditions.
   //space for lagrange multiplier 1 relate to square,
   H1_FECollection *LM1FEC = new H1_FECollection(lmOrder, 1);
   FiniteElementSpace *LM1FESpace = new FiniteElementSpace(squaremesh, LM1FEC);
   int LM1nbrDof = LM1FESpace->GetTrueVSize(); 
   cout << LM1nbrDof << " LM1FESpace degree of freedom\n";

   //space for lagrange multiplier 2 relate to round,a
   H1_FECollection *LM2FEC = new H1_FECollection(lmOrder, 1);
   FiniteElementSpace *LM2FESpace = new FiniteElementSpace(roundmesh, LM2FEC);
   int LM2nbrDof = LM2FESpace->GetTrueVSize(); 
   cout << LM2nbrDof << " LM2FESpace degree of freedom\n";   


   //The next two spaces are for application of boundary conditions.
   //space for u over the submesh 1 relate to square,
   H1_FECollection *squareFEC = new H1_FECollection(fieldOrder, 1);
   FiniteElementSpace *squareFESpace = new FiniteElementSpace(squaremesh, squareFEC);
   int squarenbrDof = squareFESpace->GetTrueVSize(); 
   cout << squarenbrDof << " squareFESpace degree of freedom\n";

   //space for lagrange multiplier 2 relate to round,
   H1_FECollection *circFEC = new H1_FECollection(fieldOrder, 1);
   FiniteElementSpace *circFESpace = new FiniteElementSpace(roundmesh, circFEC);
   int circnbrDof = circFESpace->GetTrueVSize(); 
   cout << circnbrDof << " circFESpace degree of freedom\n";   

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the
   //     Diffusion domain integrator.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the parallel bilinear form.
   a.Assemble();
   a.Finalize();

   
//
// MBLF for square boundary and LM1.
//
   MixedBilinearForm *MBLF_LM1 = new MixedBilinearForm(squareFESpace /*trial*/, LM1FESpace/*test*/);
   MBLF_LM1->AddDomainIntegrator(new MassIntegrator(one));
   MBLF_LM1->Assemble();
   MBLF_LM1->Finalize();
   SubmeshOperator *MBLF_LM1_map = new SubmeshOperator(MBLF_LM1->SpMat(), fespace, *squareFESpace, *LM1FESpace);

//
// MBLF for circular boundary and LM2.
//
   MixedBilinearForm *MBLF_LM2 = new MixedBilinearForm(circFESpace /*trial*/, LM2FESpace/*test*/);
   MBLF_LM2->AddDomainIntegrator(new MassIntegrator(one));
   MBLF_LM2->Assemble();
   MBLF_LM2->Finalize();
   SubmeshOperator *MBLF_LM2_map = new SubmeshOperator(MBLF_LM2->SpMat(), fespace, *circFESpace, *LM2FESpace);


   //
   // Linearform for LM1.
   //
   LinearForm LF_LM1(LM1FESpace);
   ConstantCoefficient minusone(-1.0);
   LF_LM1.AddDomainIntegrator(new DomainLFIntegrator(minusone));
   LF_LM1.Assemble();


   //
   // Linearform for LM2.
   //
   LinearForm LF_LM2(LM2FESpace);
   LF_LM2.AddDomainIntegrator(new DomainLFIntegrator(one));
   LF_LM2.Assemble();

   //
   // create the block structure.
   //
   Array<int> BlockOffsets(4);
   BlockOffsets[0]=0;
   BlockOffsets[1]=a.Height(); 
   BlockOffsets[2]=MBLF_LM1->Height();
   BlockOffsets[3]=MBLF_LM2->Height();
   BlockOffsets.PartialSum();
      
   {
      std::ofstream out("out/BlockOffsets.txt");
      BlockOffsets.Print(out, 10);
   }

   BlockOperator *BlockOp = new BlockOperator(BlockOffsets);

   TransposeOperator *MBLF_LM1_map_T = new TransposeOperator(MBLF_LM1_map);
   TransposeOperator *MBLF_LM2_map_T = new TransposeOperator(MBLF_LM2_map);
   

   BlockOp->SetBlock(0, 0, &(a.SpMat()));
   BlockOp->SetBlock(0, 1, MBLF_LM1_map_T);
   BlockOp->SetBlock(0, 2, MBLF_LM2_map_T);
   BlockOp->SetBlock(1, 0, MBLF_LM1_map);
   BlockOp->SetBlock(2, 0, MBLF_LM2_map);
      
   cout  << BlockOp->Height() << " BlockOp->Height()" << endl;
   cout  << BlockOp->Width() << " BlockOp->Width()" << endl;

      // the solution vector.
      Vector x(BlockOp->Height());

   //
   // the rhs vector.
   //

   BlockVector B(BlockOffsets);
   B.GetBlock(0) = b;
   B.GetBlock(1) = LF_LM1;
   B.GetBlock(2) = LF_LM2;
      

// Build the preconditioners for ...
// 
//
//                                [ M  B1^T B2^T ]
//                                [ B1   0   0   ]
//                                [ B2   0   0   ]

ThreeBlockDiagonalPreconditioner *prec;
if(precType == 1)
{
   prec = new ThreeBlockDiagonalPreconditioner(BlockOffsets, a.SpMat(), MBLF_LM1->SpMat(), MBLF_LM2->SpMat());
}

CompleteSchurPreconditioner *schur;
if(precType == 2)
{
   DSmoother *Ainv = new DSmoother(1, 1.0, 10);
   Ainv->SetOperator(a.SpMat());
   schur = new CompleteSchurPreconditioner(BlockOffsets, *Ainv, *MBLF_LM1_map, *MBLF_LM2_map);
}


      
   // 13. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use Jacobi smoothing, for now.

   StopWatch SW;
      
   if(1)
   {
      GMRESSolver solver;
      solver.SetAbsTol(0.001);
	  //solver.SetRelTol();
      solver.SetMaxIter(100000);
      solver.SetPrintLevel(1);
      solver.SetKDim(100);
      solver.SetOperator(*BlockOp);
      if(precType == 0) ; 
      else if (precType == 1) solver.SetPreconditioner(*prec);
      else if (precType == 2) solver.SetPreconditioner(*schur);
      else MFEM_VERIFY(false, "invalid preconditioner type");
      SW.Start();  
      solver.Mult(B, x);
      SW.Stop();
      cout  << solver.GetFinalRelNorm() << " GetFinalRelNorm" << endl;
      cout  << SW.UserTime() << " User Time" << endl;
      
   }

   GridFunction gf(&fespace, x, 0);
   

// 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << gf << flush;
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
