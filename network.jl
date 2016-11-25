using DataFrames

# sigmoid function and its derivative
sig(x)  = 1/(1+e^(-x))
dsig(x) = x*(1-x)

# for standardizing RGB values
in_std(x) = (1/255)*x

# returns a component-wise multiplication of two matricies of the same size
function vmult(x,y)
  if size(x) != size(y)
    return error("input lengths must match")
  else
    result = []
    for i=1:length(x)
      push!(result,x[i]*y[i])
    end
  end
  return(reshape(result,size(x)))
end

# read the table of reds:
# the first three columns are RGB values
# and the last column is boolean whether it's red or not
reds = convert(Array,readtable("reds2.csv"))
tindata = in_std(reds[:,1:3])
toutdata = (reds[:,4])

# reads in a table of two two-digit binary numbers and their sum
# this was to attempt teaching the network how to perform binary addition
# I couldn't get it to work but anyone here is welcome to try
adddata = convert(Array,readtable("adding3.csv",header=false))
addin = adddata[:,1:4]
addout = adddata[:,5:7]

# this is all one neuron does!
function neuron(ins,wts)
  x = ins * ((wts'))
  return(map(sig,x))
  #return(x)
end

# this is all the network does; the real issue is training it
function think(ins,wts)
  results = []
  push!(results,neuron(ins,wts[1]))
  #if length(wts)
  for i in 2:length(wts)
    push!(results,neuron(results[i-1],wts[i]))
  end
  return(results)
end

#=
function adj(ins,ers,outs)
  return ins' * vmult(map(dsig2,outs),ers)
  return  ins' * ers
end
=#

# creates a new network from a vector of layer sizes and the number of inputs
# for instance, if layers = [4,2,7], the first layer will have 4 neurons,
# the second layer will have 2, and the third layer will have 7
function nn_init(layers,in_size)
  println("NEW INITIATION STARTING\n\nNEURAL NETWORK WITH ",in_size," INPUTS")
  brain = []
  insert!(layers,1,in_size)
  #push!(layers,1)
  for i in 2:length(layers)
    println("LAYER ",i," HAS ",layers[i]," NEURONS")
    push!(brain,(rand(layers[i],layers[i-1])-0.5))
  end
  return(brain)
  #return(layers)
end

# component-wise multiplcation of sigmoid'(output) and output errors
delta(outs,errs) = vmult(map(dsig,outs),errs)

# computes errors and deltas for each layer recursively
# this "error delta" method is similar to a very simplified version of
# gradient descent,  and doesn't work well for networks with many layers
function ErrorDelta(ins,ans,outs,wts)
  l = length(outs)
  if l == 2
    ED = []
    E = []
    D = []
    insert!(E,1,(ans-outs[2])) #calculate errors for output neuron
    #println("E=",E)
    insert!(D,1,delta(outs[2],E[1])) #calculate delta for output neuron
    #println("D=",D)
    insert!(E,1,(D[1]*wts[2]))
    insert!(D,1,delta(outs[1],E[1]))
    push!(ED,E)
    push!(ED,D)
    #println("ED\n",ED)
    return(ED)
  else
    ED = ErrorDelta(ins,ans,outs[2:l],wts[2:l])
    insert!(ED[1],1,(ED[2][1]*wts[2]))
    insert!(ED[2],1,delta(outs[1],ED[1][1]))
    return(ED)
  end
end

# given deltas, adjust weights of network accordingly
function adjust(deltas,ins,outs)
  adj = []
  push!(adj,(ins'*deltas[1])')
  for i in 2:length(deltas)
    push!(adj,((outs[i-1])'*deltas[i])')
  end
  return(adj)
end

# train function takes in training set, layer vector, and number of iterations,
# and returns the weights of the network after adjusting each iteration
# it begins with fully random weights
function train(ins,ans,layers,inum)
  weights = nn_init(layers,size(ins)[2])
  println("\nSTARTING WEIGHTS\n",weights,"\n")
  for i in 1:inum
    #println("ITERATION ",i)
    outputs = think(ins,weights)
    #println("out size")
    #println(size(outputs))
    #println()
    errs = ErrorDelta(ins,ans,outputs,weights)
    #println("errs=")
    #println(errs)
    #println()
    adjustments = adjust(errs[2],ins,outputs)
    #println("adjustments=")
    #println(adjustments)
    #println()
    weights = weights + adjustments
  end
  return(weights)
end

# this is currently set up for the reds dataset
# a note: the last layer of neurons must be equal to the number of outputs
# so for the reds dataset, with 1 output, the last layer has 1 neuron.
wtsf = train(tindata,toutdata,[3,2,1],10000)
println("\nFINAL WEIGHTS\n",wtsf)
# input format for think: in_std([R G B])
println(think(in_std([55 27 183]),wtsf))
