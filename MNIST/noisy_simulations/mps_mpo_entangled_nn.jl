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
using BlackBoxOptim

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

  user_input = readline()
  noise = parse(Float64, user_input)

  Cvec = readdlm("Cop_4x4.txt")
  Cvec2=Cvec


  N = trunc(Int,log(size(Cvec)[1])/log(2))

  perm = zeros(Int8,2*N)

  for i in 1:N
      perm[2*i-1]=i
      perm[2*i]=i+N
  end

  # build a dense matrix for the cost function
  basis = vec(reverse.(Iterators.product(fill([0,1], N)...)))

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
      
        ten =  itensor(tt,indexnow,hilbertspace[i+1]')
        push!(Mc,ten)


      else
         U = reshape(U,(l,2,xi))
         tt = zeros(2,2,l,xi)
         @einsum tt[ii,jj,kk,ll] = U[kk,mm,ll]*ct[ii,jj,mm]
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

  # gate layer functions
  Rylayer(N, θ) = [("Ry", j, (θ = θ[j],)) for j in 1:N]
  Rzlayer(N, ϕ) = [("Rz", j, (ϕ = ϕ[j],)) for j in 1:N]
  #CXlayer(N, Π) = [("CX", (j, j+1)) for j in (1+Π):2:N-1]

  CXlayer(N, Π) = [("CX", (j, j+1)) for j in (1+Π):2:N-1]

  Hlayer(N) = [("H", j) for j in 1:N]

  # function that builds the variational circuit
  function variationalcircuit(N, depth, θ⃗)
    circuit = Tuple[]
    for d in 1:depth
      circuit = vcat(circuit, CXlayer(N, d % 2))
      circuit = vcat(circuit, Rylayer(N, θ⃗[d]))
      circuit = vcat(circuit, Rzlayer(N, θ⃗[d+depth]))
    end
    return circuit
  end

  depth = 4

  # XXX: product state currently not differentiable, which means
  # it has to stay outside
  ψ = productstate(hilbertspace)

  # noise model
  noisemodel = (1 => ("depolarizing", (p = 0.001*noise,)),
                2 => ("depolarizing", (p = 0.00375*noise,)))

  # compute the loss function
  function loss(θ⃗)
     # theta comes from the blackbox optimizer in its vector notation
     # We need to convert theta to an array readable by circuit constructor (a vector of vectors) 
     A = reshape(θ⃗, (depth*2,N)) 
     θ⃗x =[A[i,:] for i in 1:size(A,1)] 

     circuit = variationalcircuit(N, depth, θ⃗x)
   
     Uψ = runcircuit(hilbertspace,circuit;noise=noisemodel,cutoff=1e-18)
  
     return real(inner(Uψ,C)) 
  end


  # sample quantum state Ns times
  function sample(θ⃗,Ns)
     # theta comes from the blackbox optimizer in its vector notation
     # We need to convert theta to an array readable by circuit constructor (a vector of vectors)
     A = reshape(θ⃗, (depth*2,N))
     θ⃗x =[A[i,:] for i in 1:size(A,1)]

     circuit = variationalcircuit(N, depth, θ⃗x)
     
     Uψ = runcircuit(hilbertspace,circuit;noise=noisemodel,cutoff=1e-18)
     S = getsamples(Uψ, Ns) 
     return S 
  end

  hist=[]
  param = []
  for ii in 1:10 # 10 optimization runs
  print(ii)
  # initial parameters
  Random.seed!()
  θ⃗₀ = [2π .* rand(N) for _ in 1:depth*2]

  # best for these circuits: :probabilistic_descent algorithm 
  res = bboptimize(loss; SearchRange = (-π, π), NumDimensions = 2*depth*N, Method= :probabilistic_descent, MaxSteps=2000 )
 
   
  θ⃗ = best_candidate(res)
  loss_s = best_fitness(res) 
  Ns=1000
  S = sample(θ⃗,Ns)

  av=0.0
  av2=0.0
  aa = zeros(Ns) 
  for jj in 1:Ns
     binary_string = reverse(join(string.(S[jj,:]))) # reverse(join(string.(S[jj,:])))
     decimal_integer = parse(Int, binary_string, base=2)
     av=av+Cvec2[decimal_integer+1]
     av2=av2+Cvec2[decimal_integer+1]^2
     aa[jj]=Cvec2[decimal_integer+1] 
  end
  av2=av2/Ns
  av=av/Ns
  #Saving the energies of each sample and averages
  println("average = ", av," pm ", sqrt(abs((av)^2-av2)/Ns))
  writedlm("loss_samples"*string(ii, base = 10)*".txt", aa)

  a = reduce(vcat, θ⃗)

  push!(hist,loss_s)
  push!(a,loss_s)


  push!(param,a)



  end



writedlm("histogram.txt",hist, "   ")
writedlm("params.txt",param, "   ")


end
