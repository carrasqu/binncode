using PastaQ
using ITensors
using Test
using Random
using Printf
using OptimKit
using Zygote
using DelimitedFiles
using LinearAlgebra
using DelimitedFiles
using Einsum


function tensor_to_mpo(T::ITensor)
  M = ITensor[]
  N = PastaQ.nsites(T)
  for k in 1:N-1
    sinds = k == 1 ? inds(T, tags = "n=$k") :
                     (commoninds(T, M[k-1])...,inds(T, tags = "n=$k")...)
    print(" sinds ",sinds," \n")
    u,s,T = svd(T, sinds...)
    #print("---- \n", u,s,T, "-----\n")
    push!(M, u*s)
    if k == N-1
      push!(M,T)
      break
    end
  end
  return MPO(M)
end




let


  Cvec = readdlm("Cop_4x4.txt")


  N = trunc(Int,log(size(Cvec)[1])/log(2))

  perm = zeros(Int8,2*N)

  for i in 1:N
      perm[2*i-1]=i
      perm[2*i]=i+N
  end

  # build a dense matrix for the cost function
  basis = vec(reverse.(Iterators.product(fill([0,1], N)...)))
#   Cmat = zeros(fill(2,2*N)...)
#   Cmatj = zeros(fill(2,2*N)...)
#   s = zeros(Int8,2*N)
#   for (i,σ) in enumerate(basis)
#     s = Tuple(permute!([(σ.+1)...,(σ.+1)...],perm))
#     Cmat[(σ.+1)...,(σ.+1)...] = Cvec[i]     #
#                                             # 1 3 5 7
#     Cmatj[s[1:N]...,s[N+1:2*N]...]=Cvec[i]  # o-o-o-o
#   end                                       # 2 4 6 8

  xi = 4
  l=1
  indexprev = Index(l; tags = "Link,v", plev = 0)
  Cmps = []
  M = ITensor[]
  Mc = ITensor[]
  hilbertspace = qubits(N)

  ct = zeros(2,2,2)
  ct[1,1,1] =1
  ct[2,2,2] =1
  #ct = reshape(ct,(4,2))

  for i in 1:N-1

      Cmatj = reshape(Cvec,(2*l,2^(N-i)))
      U,S,Vt = svd(Cmatj)

      F = svd(Cmatj)
      xi = length(S[S .> 10^(-10)]) # "compress"
      S = F.S[1:xi]
      U = F.U[:,1:xi]*diagm(S)

      Cvec = F.Vt[1:xi,:]

      push!(Cmps, reshape(U,(l,2,xi)))


      indexnow = Index(xi; tags = "Link,v", plev = 0)

      if i == 1
        tt = zeros(2,2,xi)
        @einsum tt[ii,jj,ll] = ct[ii,jj,mm]*U[mm,ll]
        ten =  itensor(tt,hilbertspace[i]',hilbertspace[i],indexnow)
        push!(M, ten)

        tt = zeros(2,xi)
        tt = U
        ten =  itensor(tt,hilbertspace[i]',indexnow)
        push!(Mc,ten)

      elseif i==N-1
         U = reshape(U,(l,2,xi))
         tt = zeros(2,2,l,xi)
         @einsum tt[ii,jj,kk,ll] = U[kk,mm,ll]*ct[ii,jj,mm]

         ten = itensor(tt,hilbertspace[i]',hilbertspace[i],indexprev,indexnow)
         push!(M, ten)
         tt = zeros(2,2,xi)
         @einsum tt[ii,jj,ll] = Cvec[ll,mm]*ct[ii,jj,mm]
         ten = itensor(tt,hilbertspace[i+1]',hilbertspace[i+1],indexnow)
         push!(M, ten)

        tt = zeros(l,2,xi)
        tt = U
        ten =  itensor(tt,indexprev,hilbertspace[i]',indexnow)
        push!(Mc,ten)

        tt = zeros(xi,2)
        tt = Cvec
        #print(size(Cvec)," ",xi)
        ten =  itensor(tt,indexnow,hilbertspace[i+1]')
        push!(Mc,ten)


         #push!(M, ten)
      else
         U = reshape(U,(l,2,xi))
         tt = zeros(2,2,l,xi)
         @einsum tt[ii,jj,kk,ll] = U[kk,mm,ll]*ct[ii,jj,mm]
         #tope = reshape(U,(l,2,2,xi))
         #pp = [2,3,1,4]
         #tt = permutedims(tope,pp)
         ten = itensor(tt,hilbertspace[i]',hilbertspace[i],indexprev,indexnow)
         push!(M, ten)

        tt = zeros(2,xi,2)
        tt = U
        ten =  itensor(tt,indexprev,hilbertspace[i]',indexnow)
        push!(Mc,ten)
      end

      l = xi
      indexprev = indexnow

  end
  C = MPO(M)
  #print(C)

#   allc =  reverse.(Iterators.product(fill(1:2,N)...))[:]
#   Ca = diag(reshape( array(C[1]*C[2]*C[3]*C[4]*C[5]), (2^N,2^N)))
#   print(Ca)

  #return
#   Cmps = MPS(Mc)
#   print(reshape(array(Cmps[1]*Cmps[2]*Cmps[3]*Cmps[4]*Cmps[5]),(2^5)))
#   print(" ", )
 # return
  #print(Cj)
  #print(size(Cmpo)[1])

  # define the hilbert space
  #hilbertspace = qubits(N)
  # build the tensor for the cost function
  #Ctensor = itensor(Cmat, prime.(hilbertspace)..., hilbertspace...)
  # build MPO decomposition
  #C = tensor_to_mpo(Ctensor)
  #print(C)
  #return

  # gate layer functions
  Rylayer(N, θ) = [("Ry", j, (θ = θ[j],)) for j in 1:N]
  Rzlayer(N, ϕ) = [("Rz", j, (ϕ = ϕ[j],)) for j in 1:N]
  #CXlayer(N, Π) = [("CX", (j, j+1)) for j in (1+Π):2:N-1]

  CXlayer(N, Π) = [("CX", (j, j+1)) for j in (1+Π):2:N-1]

  Hlayer(N) = [("H", j) for j in 1:N]

  # function that builds the variational circuit
  function variationalcircuit(N, depth, θ⃗)
    circuit = Tuple[]
    #circuit = vcat(circuit, Hlayer(N))
    for d in 1:depth
      circuit = vcat(circuit, CXlayer(N, d % 2))
      circuit = vcat(circuit, Rylayer(N, θ⃗[d]))
      circuit = vcat(circuit, Rzlayer(N, θ⃗[d+depth]))
    end
    return circuit
  end

  depth = 6

  # XXX: product state currently not differentiable, which means
  # it has to stay outside
  ψ = productstate(hilbertspace)

  # cost function
  function loss(θ⃗)
    circuit = variationalcircuit(N, depth, θ⃗)
    U = buildcircuit(hilbertspace, circuit)
    return rayleigh_quotient(C, U, ψ)
  end

  hist=[]
  param = []
  for ii in 1:10
  print(ii)
  # initial parameters
  Random.seed!()
  θ⃗₀ = [2π .* rand(N) for _ in 1:depth*2]

  @printf("Initial cost function: ⟨C⟩ = %.8f\n",loss(θ⃗₀))

  optimizer = LBFGS(verbosity=2,gradtol = 1e-6,maxiter = 500 )
  loss_n_grad(x) = (loss(x), convert(Vector, loss'(x)))
  θ⃗, fs, gs, niter, normgradhistory = optimize(loss_n_grad, θ⃗₀, optimizer)

  #circuit = variationalcircuit(N, depth, θ⃗)
  #ψo = runcircuit(circuit)
  #ψvec = PastaQ.array(ψo)
#  for (i,σ) in enumerate(basis)
#      if abs(ψvec[i])>10^(-5)#
#         print("⟨",σ...,"|ψ⟩ = ")
#         @printf("%.5f\n",ψvec[i])
      #end
  a = reduce(vcat, θ⃗)
  loss_s = loss(θ⃗)
  push!(hist,loss_s)
  push!(a,loss_s)
  #print("losss",loss(θ⃗))

  push!(param,a)
  #print(param)

  #end
  end

#print(hist)
#print(param)
writedlm("histogram.txt",hist, "   ")
writedlm("params.txt",param, "   ")


end
