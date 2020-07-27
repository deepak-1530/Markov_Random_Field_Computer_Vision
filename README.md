# Markov_Random_Field_Computer_Vision
->Using markov random fields for noise removal.

->Pott's model is used with 16 point neighbourhood. 

->Energy minimization is done using Simulated Annealing


# Unary and Pairwise potential calculation ->

pE = w*(abs(im[k,l] - im[i,j]))**x  (16 point neighbourhood is used)

totalPairWiseEnergy = w1*pE + w2*local_variance + w2*(-log(im[i,j])) (using normalized values in the image)

