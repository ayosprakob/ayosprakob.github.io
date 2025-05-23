
import itertools as itt
import re
from collections import defaultdict

# To be imported from HTML
irrep_set = []
axis_info = {}
connection_info = {}

def main():

	print('''
    This Python script is generated with the Armillary sphere builder.
    URL: https://ayosprakob.github.io/armillary_interface
       by Atis Yosprakob
	''')

	print("    --- Input ----------------------------------------------------------------")
	print()
	print("    axis_info:")
	for axis in axis_info:
		print("        ",axis,":",axis_info[axis])
	print()
	print("    connection_info:")
	for connection in connection_info:
		print("        ",connection,":",connection_info[connection])
	print()
	print("    irrep_set:")
	for iset,the_set in enumerate(irrep_set):
		print("        ",iset+1,":",the_set)
	print()
	print()

	compute_CG_tensors()
	compute_vertex_tensors() # Possibly can be improved with multiprocessing


vertex_tensors = {}
def compute_vertex_tensors():
	# Match a pair of Uμ and Vμ
	vertex_pairs = [ (key,key.replace("U","V"),key.replace("U","")) for key in CG_tensors if "U" in key ]
	template_indices = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZαβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩабвгдежзийклмнопрстуфхцчшщыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЬЭЮЯאבגדהוזחטיךכלםמןנסעףפץצקרשתאבגדהוזחטיכלמנסעפצקרשת१२३४५६७८९०१२३४५६७८९۰١٢٣٤٥٦٧٨٩٠"

	# count the pairs
	contractable_pair = []
	progress_text = ""
	ipair = 0
	for Uμ,Vμ,μ in vertex_pairs:
		# μ is a string!
		key = μ
		vertex_tensors[key] = []
		for Uelem in CG_tensors[Uμ]:
			for Velem in CG_tensors[Vμ]:
				# incices = [pre,[r,m]]
				Uindices = Uelem[:-1]
				Vindices = Velem[:-1]
				Ur = Uindices[-1][:-1]
				Vr = Vindices[-1][:-1]

				if Ur!=Vr :
					continue

				if ipair%10==0 :
					progress_text = "Estimating progress "+print_idle(ipair,58)
					print(progress_text, end="\r")
				ipair += 1

				contractable_pair.append([Uelem,Velem,key])
	print(" "*len(progress_text), end="\r")

	ipair = 0
	npair = len(contractable_pair)
	progress_text = ""
	for Uelem,Velem,key in contractable_pair:

		if ipair%2==0 :
			progress_text = "Computing vertex tensors "+print_bar(ipair,npair,53)
			print(progress_text, end="\r")
		ipair += 1

		Uindices = Uelem[:-1]
		Vindices = Velem[:-1]
		Upre = Uindices[:-1]
		Vpre = Vindices[:-1]
		Ur = Uindices[-1][:-1]
		Vr = Vindices[-1][:-1]
		Um = Uindices[-1][-1]
		Vm = Vindices[-1][-1]

		nrep_index = [Upre,Vpre,Vr,Um,Vm]
		CGU = Uelem[-1]
		CGV = Velem[-1]
		
		
		einsum1 = template_indices[:len(Upre)]
		einsumx = template_indices[len(Upre)]
		einsum2 = template_indices[(len(Upre)+1):(len(Upre)+1+len(Vpre))]
		einsum_str = einsum1+einsumx+","+einsum2+einsumx+"->"+einsum1+einsum2

		prod = contract_inner(CGU.coeffs,CGV.coeffs,einsum_str)
		shape = (*CGU.shape[:len(Upre)],*CGV.shape[:len(Vpre)])
		prod = SparseCoeff(prod,shape,nrep_index)

		vertex_tensors[key].append([*nrep_index,prod])
	#print(" "*len(progress_text), end="\r")
	print()
	
CG_tensors = {}
def compute_CG_tensors():

	# ::::: Notation and Nomenclature ::::::::::::::::::::::::::::::::::
	#  
	#  This will return a dictionary
	#  CG_tensors[<axis>] where <axis> is the key of <axis_info>
	#  The object T = CG_tensors[<axis>] is a product of
	#  CG coefficients on this particular <axis>.
	#
	#  Recall that UU...U = (CC...C)U(CC...C),
	#  The object T=CC...C on both sides are the same, and that's
	#  exactly equal to T = CG_tensors[<axis>] above.
	#
	#  Now note that the object T here is a custom object <SparseCoeff>
	#  which has following attributes/properties
	#
	#  - T.indices is a list of irreps indices [r1,r2,..,rn,ρ]
	#                                 (ρ has the multiplicity index attached)
	#  - T.coeffs is a list of nonzero components of the form (i1,i2,...,in,value)
	#
	#  - T.shape is the full shape of the tensor
	#
	#  - T.size is the volume of the tensor (number of all components)
	#
	#  - T.nnz is the number of nonzero components
	#
	#  If you want to do contraction, match the irreps first, then contract
	#  the matrix indices.
	#

	subvertex_sig = [] # the 'signature' of all unique subvertices
	subvertex_num = {} # refer to which signature each subvertex has
	cg2_list = {} # a list of CG2 signatures to be computed
	cgs = {}      # a list of all CGs separated indexed by path
	paths = {}    # a list of paths indexed by subvertex_sig (used as the indices of subvertex tensors)

	def get_C_from_axis(axis):
		signature = subvertex_sig[subvertex_num["U0"]]
		C = [ (*cgs[path].indices,cgs[path]) for path in paths[str(signature)] ]
		return C

	# Step 1: check if there are identical subvertices
	
	for key in axis_info:
		signature = tuple([ irrep_set[term] for term in axis_info[key] ])
		if signature not in subvertex_sig:
			subvertex_sig += [signature]
		subvertex_num[key] = subvertex_sig.index(signature)

	# From now on, the calculations are done per signature

	for signature in subvertex_sig:

		paths[str(signature)] = []

		combi_list = list(itt.product(*signature))
		for combi in combi_list:
			paths_current = compute_decomposition_paths([Weight(w) for w in combi])
			paths[str(signature)] += paths_current
			
			# count all cg decomposition seen so far (all signature, all combi)
			cg2_list = merge_cg2_list(cg2_list,count_cg2s(paths_current))

			stripped_path_count = {}
			for path in paths_current:
				stripped_path = re.sub(r"\([^()]*\)", "", path) # remove multiplicity index
				if stripped_path in stripped_path_count:
					stripped_path_count[stripped_path] += 1
				else:
					stripped_path_count[stripped_path]  = 1

				indices = [ s.split(" -> ")[0] for s in stripped_path.split(" x ")]
				indices += [stripped_path.split(" -> ")[-1]]
				# convert to a proper list
				indices = [ list(map(int, re.findall(r"-?\d+", w))) for w in indices ]
				total_mult = stripped_path_count[stripped_path]
				indices[-1] = [*indices[-1],total_mult-1]
				if path not in cgs:
					cgs[path] = [None,indices] # To be stored by the corresponding CG coefficient

	# count the number of required coeffs
	cg_count = 0
	DecompInfo = []
	progress_text = ""
	for signature in cg2_list :
		# convert the signature into 2 weights
		w1,w2 = signature.split(" x ")
		W1,W2 = [ Weight(list(map(int, re.findall(r"-?\d+", w)))) for w  in signature.split(" x ") ]
		decomp = Decomposition(W1,W2)
		for W3 in decomp:
			w3 = str(list(W3.elem[:-1]))
			mult = decomp.multiplicity(W3)
			for m in range(1,mult+1):

				if cg_count%10==0 :
					progress_text = "Estimating progress "+print_idle(cg_count,58)
					print(progress_text, end="\r")
					
				DecompInfo.append([(W3,W1,W2,m-1),(w1,w2,w3,m)])
				cg_count+=1
	ncgs = cg_count
	print(" "*len(progress_text), end="\r")

	# compute all cg2s
	cg2s = {}
	cg_count = 0
	progress_text = ""
	for x1,x2 in DecompInfo:

		if cg_count%2==0 :
			progress_text = "Computing CGs "+print_bar(cg_count,ncgs)
			print(progress_text, end="\r")

		CGi = Coefficients(*x1).tensor
		cg2s[x2] = CGi
		cg_count+=1
	#print(" "*len(progress_text), end="\r")
	print()

	# compute the actual cgs
	for path in cgs:
		# reformat the paths into separate CG2
		def get_subcg2_info(path_text):
			parts = re.split(r'\s*->\s*', path_text)  # Split by '->'
			temp = [f"{parts[i]} -> {parts[i+1].split(' x ')[0]}" for i in range(len(parts)-1)]
			ret = []
			for elem in temp:
				temp2 = elem.split(' x ')
				temp2[0] = re.sub(r"\([^()]*\)", "", temp2[0])
				ret.append(' x '.join(temp2))
			return ret

		subcg2s = get_subcg2_info(path)
		CG2s = [] # a list of coeff matrices to be multiplied
		CGshape = []
		indices = []
		for subcg2 in subcg2s :
			cg2_sig, result = subcg2.split(" -> ")

			w1,w2 = cg2_sig.split(" x ")
			w3 = result.split("(")[0]
			m3 = int(result.split("(")[1].replace(")",""))

			W1,W2,W3 = [ Weight(list(map(int, re.findall(r"-?\d+", w)))) for w in [w1,w2,w3] ]
			d1,d2,d3 = [ W.dimension for W in [W1,W2,W3] ]

			if len(CGshape)==0 :
				CGshape+=[d1,d2,d3]
			else:
				CGshape = CGshape[:-1]+[d2,d3]

			coeffs = cg2s[(w1,w2,w3,m3)]
			CG2s.append(coeffs)
		CGshape = tuple(CGshape)
		
		indices = cgs[path][1]
		cgs[path] = SparseCoeff(contract_CGs(CG2s),CGshape,indices)

	# Final result:

	global CG_tensors

	CG_tensors = {}
	for key in axis_info:
		CG_tensors[key] = get_C_from_axis(key)

def print_idle(i,length=64,speed=1):
	x = int(np.ceil(length*(0.5-0.5*np.cos(speed*1e-5*i))))
	x = max(1,min(x,length))
	t1 = x-1
	t2 = length-x
	text = "·"*t1+"█"+"·"*t2
	return text

def print_bar(i,n,length=64):
	nblock = min(int(np.ceil(1.0*length*i/(n-1))),length)
	nblank = length-nblock
	text = "█"*nblock+"·"*nblank
	percent = str(np.floor(1000*i/n)/10)+"%"

	pc_len = len(percent)
	if nblank>pc_len :
		text = text[:-(pc_len)]+percent
	else:
		percent = percent[(pc_len-nblank):]
		text = text.replace("·","")+percent
	return text

def compute_decomposition_paths(irreps):
    """
    Given a list of irreps [w1, w2, ..., wn], compute the decomposition
    of their product, accounting for multiplicities, and return a list of 
    (final_irrep, path) tuples.
    
    The path string format follows:
    w1 x w2 -> u1[m1] x w3 -> u2[m2] x ... x wn -> un[mn]
    """
    if not irreps:
        return []

    # Initialize with the first irrep and its trivial path.
    paths = [(irreps[0], str(irreps[0]))]

    # Loop over the remaining irreps in the product.
    for w in irreps[1:]:
        new_paths = []
        for current_irrep, branch_path in paths:
            # Decompose the product of the current result and the next irrep w.
            decomp = Decomposition(current_irrep, w)
            for candidate in decomp:
                # Get the multiplicity of this candidate.
                mult = decomp.multiplicity(candidate)
                # Create separate branches for each occurrence due to multiplicity.
                for m in range(1, mult + 1):
                    new_path = f"{branch_path} x {w} -> {candidate}({m})"
                    new_paths.append((candidate, new_path))
        paths = new_paths  # Update paths for the next iteration.

    return [ path[1] for path in paths ]

# count the muliplicity of similar components (paths = all components)
def count_irreducible_components(paths):
	multiplicity = {}
	for path in paths:
		component = path.split("->")[-1].split("(")[0]
		if component in multiplicity:
			multiplicity[component] += 1
		else:
			multiplicity[component] = 1
	return multiplicity

# count which cgs we need to compute
def count_cg2s(paths):
	cg2_list = {}
	for path in paths:
		path = re.sub(r"\([^()]*\)", "", path) # remove multiplicity index
		prods = path.split(" -> ")[:-1]
		for prod in prods:
			if prod in cg2_list:
				cg2_list[prod] += 1
			else:
				cg2_list[prod] = 1
	return cg2_list

def merge_cg2_list(list1,list2):
	# hint: they are both dictionary...
	ret = {}
	for key in list1:
		ret[key] = list1[key]
	for key in list2:
		if key in ret:
			ret[key] += list2[key]
		else:
			ret[key] = list2[key]
	return ret

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Below here are the computations. Do not touch unless you know what you are doing.
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

'''

  This is the Python script converted from Arne Alex's C++ code provided in
  https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/ClebschGordan.cpp
  This file contains everything in the namespace "clebsch"
               
  Thanks, Arne.

  I also compiled this into the WebAssembly version in https://ayosprakob.github.io/clebsch

  -- Converted by Atis Yosprakob // 3 Mar 2025

'''

import numpy as np
from numpy.linalg import svd
from numpy.linalg import lstsq

from bisect import bisect_left


EPS = 1e-12

# ==== Binomial ===================================================================================
#       Store binomial function in cache for fast computation of the binomial coefficients

class Binomial:

    def __init__(self):
        self.cache = []
        self.N = 0
    
    def __call__(self, n, k):
        if self.N <= n:
            new_size = (n + 1) * (n + 2) // 2
            if len(self.cache) < new_size:
                self.cache.extend([0] * (new_size - len(self.cache)))
            
            while self.N <= n:
                index = self.N * (self.N + 1) // 2
                self.cache[index] = 1
                self.cache[index + self.N] = 1
                
                for j in range(1, self.N):
                    self.cache[index + j] = (
                        self.cache[(self.N - 1) * self.N // 2 + j] +
                        self.cache[(self.N - 1) * self.N // 2 + j - 1]
                    )
                
                self.N += 1
        return self.cache[n * (n + 1) // 2 + k]
binomial = Binomial()


# ==== Weight =====================================================================================
#       This is the weight vector w = (n1,n2,...,nN-1,0)

class Weight:

    def __init__(self, N, index=None):

        # This is a hidden constructor that the original uses but it's somehow never declared
        if isinstance(N,Weight):
            # cannot simply use the copy method
            other = N
            self.N = other.N
            self.elem = other.elem.copy()
        elif isinstance(N, (list, tuple, np.ndarray)):
            weight = tuple(N)
            N = len(weight)
            if use_N_weight_notation :
                assertion(weight[-1]==0,"The last component of the weight has to be zero in the N-weight notation.")
            else:
                N += 1
                weight = (*weight,0)
            self.N = N
            self.elem = np.array(weight)
        else:
            self.N = N
            self.elem = np.zeros(N, dtype=int)

            if index!=None :
                for i in range(N):
                    if index<=0 :
                        break
                    j=1
                    while binomial(N-i-1+j, N-i-1)<= index :
                        self.elem[i] = j
                        j<<=1

                    j = self.elem[i]>>1
                    while j>0 :
                        if binomial(N-i-1+(self.elem[i]|j),N-i-1)<=index:
                            self.elem[i]|=j
                        j >>= 1
                    index -= binomial(N-i-1+self.elem[i], N-i-1)
                    self.elem[i] += 1

    def __repr__(self):
        return str(list(self.elem[:self.N if use_N_weight_notation else self.N-1]))

    def __getitem__(self, k):
        assertion(k>=1 or k<=self.N ,"Weight component's index must be between 1 and N.")
        return self.elem[k-1]
    
    def __setitem__(self, k, value):
        assertion(k>=1 or k<=self.N ,"Weight component's index must be between 1 and N.")
        self.elem[k-1] = value

    def __lt__(self, other):
        assertion(self.N == other.N ,"The two weights must be of the same length.")
        for i in range(self.N):
            if self.elem[i]-self.elem[self.N-1] != other.elem[i]-other.elem[other.N-1]:
                return self.elem[i]-self.elem[self.N-1] < other.elem[i]-other.elem[other.N-1]
        return False

    def __eq__(self, other):
        if other==None:
            return False
        assertion(self.N == other.N ,"The two weights must be of the same length.")
        for i in range(self.N):
            if self.elem[i]-self.elem[i-1] != other.elem[i]-other.elem[i-1]:
                return False
        return True

    def __add__(self, other):
        assertion(self.N == other.N ,"The two weights must be of the same length.")
        result = Weight(self.N)
        result.elem = self.elem+other.elem
        return result

    def copy(self):
        result = Weight(self.N)
        result.elem = self.elem.copy()
        return result

    @property
    def index(self):
        result = 0
        i=0
        while self.elem[i]>self.elem[self.N-1] :
            result += binomial(self.N-i-1+self.elem[i]-self.elem[self.N-1]-1,self.N-i-1)
            i+=1
        return result

    @property
    def dimension(self):
        numerator = 1
        denominator = 1
        for i in range(1,self.N):
            for j in range(self.N-i):
                numerator *= self.elem[j] - self.elem[i + j] + i
                denominator *= i
        return numerator // denominator

# ==== Pattern ====================================================================================
#       Gelfand-Tsetlin pattern

class Pattern:

    def __init__(self, *args):
        self.N = None
        self.elem = None

        pattern = None
        weight = None
        index = None
        if len(args) == 1 :
            if isinstance(args[0],Pattern) :
                pattern = args[0]
            elif isinstance(args[0],Weight):
                weight = args[0]
                index = 0
        elif len(args) == 2 :
            weight, index = args
        else:
            raise ValueError("Too many arguments.")

        if pattern != None :
            self.N = pattern.N
            self.elem = pattern.elem.copy()
        elif weight!=None and index !=None:
            self.N = weight.N
            self.elem = np.zeros(self.N*(self.N+1)//2, dtype=int)

            for i in range(1,self.N+1):
                self[i,self.N] = weight[i]

            for l in range(self.N - 1, 0, -1):
                for k in range(1, l + 1):
                    self[k,l] = self[k+1,l+1]

            while index > 0:
                index -= 1
                assertion(self.increment ,"The index must not exceed the weight's dimension.")

    def __repr__(self):
        # this is my own function, not the original one
        result = ""
        index = 0
        for i in range(self.N):
            if i>0:
                result+="\n"
            result+="[ "+" "*i

            for j in range(self.N-i):
                if j>0:
                    result+=" "
                result += str(self.elem[index])
                index+=1
            result+=" "*i+" ]"

        return result

    def __getitem__(self, indices):
        k,l = indices
        return self.elem[ (self.N*(self.N+1)-l*(l+1))//2+k-1 ]

    def __setitem__(self, indices, value):
        k,l = indices
        self.elem[ (self.N*(self.N+1)-l*(l+1))//2+k-1 ] = value

    def __iadd__(self, num):
        assertion(isinstance(num,int) and num>=0)
        result = True
        for i in range(num) :
            result = result and self.increment
        return self

    def __isub__(self, num):
        assertion(isinstance(num,int) and num>=0)
        result = True
        for i in range(num) :
            result = result and self.decrement
        return self

    @property
    def increment(self):
        k=1
        l=1
        while( l<self.N and self[k,l]==self[k,l+1]):
            k -= 1
            if k==0 :
                l += 1
                k = l

        if l==self.N :
            return False

        self[k,l] = self[k,l]+1

        while k!=1 or l!=1 :
            k+=1
            if k>l :
                k=1
                l-=1

            self[k,l] = self[k+1,l+1]

        return True

    @property
    def decrement(self):
        k=1
        l=1
        while( l<self.N and self[k,l]==self[k+1,l+1]):
            k -= 1
            if k==0 :
                l += 1
                k = l

        if l==self.N :
            return False

        self[k,l] = self[k,l]-1

        while k!=1 or l!=1 :
            k+=1
            if k>l :
                k=1
                l-=1

            self[k,l] = self[k,l+1]

        return True

    @property
    def index(self):
        result = 0
        p = Pattern(self)
        while p.decrement :
            result += 1
        return result

    def get_weight(self):
        result = Weight(self.N)
        prev = 0
        for l in range(1,self.N+1):
            now=0
            for k in range(1,l+1):
                now+=self[k,l]

            result[l] = now-prev
            prev=now

        return result

    def lowering_coeff(self, k, l):
        result = 1.0
        for i in range(1,l+2):
            result *= self[i,l+1]-self[k,l]+k-i+1
        for i in range(1,l):
            result *= self[i,l-1]-self[k,l]+k-i
        for i in range(1,l+1):
            if i== k:
                continue
            result /= self[i,l]-self[k,l]+k-i+1
            result /= self[i,l]-self[k,l]+k-i
        return np.sqrt(-result)

    def raising_coeff(self, k, l):
        result = 1.0
        for i in range(1,l+2):
            result *= self[i,l+1]-self[k,l]+k-i
        for i in range(1,l):
            result *= self[i,l-1]-self[k,l]+k-i-1
        for i in range(1,l+1):
            if i== k:
                continue
            result /= self[i,l]-self[k,l]+k-i
            result /= self[i,l]-self[k,l]+k-i-1
        return np.sqrt(-result)

# ==== Decomposition ==============================================================================
#       Computation of the decomposition of r1 x r2 into components with multiplicities

class Decomposition:

    def __init__(self, factor1, factor2):
        self.N = factor1.N
        assertion(factor1.N == factor2.N)
        self.weights = []
        self.multiplicities = []

        result = []
        low = Pattern(factor1)
        high = Pattern(factor1)
        trial = Weight(factor2)
        k = 1
        l = self.N

        ix = 0
        while True:
            while k <= self.N:
                l -= 1
                if k <= l:
                    low[k, l] = max(high[k + self.N - l, self.N], high[k, l + 1] + trial[l + 1] - trial[l])
                    high[k, l] = high[k, l + 1]
                    if k > 1 and high[k, l] > high[k - 1, l - 1]:
                        high[k, l] = high[k - 1, l - 1]
                    if l > 1 and k == l and high[k, l] > trial[l - 1] - trial[l]:
                        high[k, l] = trial[l - 1] - trial[l]
                    if low[k, l] > high[k, l]:
                        break
                    trial[l + 1] += high[k, l + 1] - high[k, l]
                else:
                    trial[l + 1] += high[k, l + 1]
                    k += 1
                    l = self.N

            if k > self.N:
                result.append(trial.copy())
                for i in range(1, self.N + 1):
                    result[-1][i] -= result[-1][self.N]
            else:
                l += 1

            while k != 1 or l != self.N:
                if l == self.N:
                    k-=1
                    l = k - 1
                    trial[l + 1] -= high[k, l + 1]
                elif low[k, l] < high[k, l]:
                    high[k, l] -= 1
                    trial[l + 1] += 1
                    break
                else:
                    trial[l + 1] -= high[k, l + 1] - high[k, l]
                l += 1

            if k == 1 and l == self.N:
                break

            ix += 1
        result.sort()

        for res in result:
            if result.index(res) != 0 and res == self.weights[-1]:
                self.multiplicities[-1] += 1
            else:
                self.weights.append(res.copy())
                self.multiplicities.append(1)

    def __repr__(self):
        result = ""
        i = 0
        for weights,multiplicities in zip(self.weights,self.multiplicities):
            if i>0 :
                result+="\n"+"  + "
            else:
                result+="  "
            result += str(weights)+" × "+str(multiplicities)
            i+=1
        return result

    @property
    def size(self):
        return len(self.weights)

    def __getitem__(self, j):
        return self.weights[j]

    def __setitem__(self, j, value):
        self.weights[j] = value

    def multiplicity(self, irrep):
        assertion(irrep.N == self.N)
        index = bisect_left(self.weights, irrep)
        return self.multiplicities[index] if index < self.size and self.weights[index] == irrep else 0

# ==== Coefficients ===============================================================================
#       The CG coefficients "clzx" which is a dict of nonzero elements

class Coefficients:

    def __init__(self, irrep, factor1, factor2, mult=-1):
        
        # mult is an optional extra factor that I place myself
        # to specify which multiplicity index we want to see.
        # Not to be confused with self.multiplicity

        self.clzx = {}
        self.N = irrep.N
        self.irrep = irrep
        self.factor1 = factor1
        self.factor2 = factor2
        self.irrep_dimension = irrep.dimension
        self.factor1_dimension = factor1.dimension
        self.factor2_dimension = factor2.dimension
        self.multiplicity = Decomposition(factor1,factor2).multiplicity(irrep)
        self.multiplicity_index = mult

        assertion(factor1.N == irrep.N)
        assertion(factor2.N == irrep.N)

        self.compute_highest_weight_coeffs()

        for i in range(self.multiplicity):
            done = [False] * self.irrep_dimension
            done[self.irrep_dimension - 1] = True
            for j in range(self.irrep_dimension - 1, -1, -1):
                if not done[j]:
                    self.compute_lower_weight_coeffs(i, j, done)

    def __repr__(self):

        coeff_entries = {}
        #if self.multiplicity_index > 0 :
        #    for key in self.clzx:
        #        if key[2]==self.multiplicity_index :
        #            coeff_index = (key[0],key[1],key[3],key[2])
        #            coeff_entries[coeff_index] = self.clzx[key]
        #else:
        #    for key in self.clzx:
        #        coeff_index = (key[0],key[1],key[3],key[2])
        #        coeff_entries[coeff_index] = self.clzx[key]

        for key in self.clzx:
            coeff_entries[key] = self.clzx[key]

        coeff_entries = {key: coeff_entries[key] for key in sorted(coeff_entries)}

        result = ""
        for key in coeff_entries:
            if np.abs(coeff_entries[key])>EPS:
                result += str(key)+" : "+str(coeff_entries[key])+"\n"
        return result

    @property
    def tensor(self):

        assertion(self.multiplicity_index > -1, "enter the multiplicity index")

        result = []
        for key in self.clzx:
            if key[2]==self.multiplicity_index :
                i1 = key[0]
                i2 = key[1]
                i  = key[3]
                result += [(i1,i2,i,self.clzx[key])]

        return result

    def get(self, factor1_state, factor2_state, multiplicity_index, irrep_state):
        assertion(0 <= factor1_state and factor1_state < self.factor1_dimension)
        assertion(0 <= factor2_state and factor2_state < self.factor2_dimension)
        assertion(0 <= multiplicity_index and multiplicity_index < self.multiplicity)
        assertion(0 <= irrep_state and irrep_state < self.irrep_dimension)

        coefficient_label = (factor1_state, factor2_state, multiplicity_index, irrep_state)
        return self.clzx.get(coefficient_label, 0)

    def set(self, factor1_state, factor2_state, multiplicity_index, irrep_state, value):
        assertion(0 <= factor1_state and factor1_state < self.factor1_dimension)
        assertion(0 <= factor2_state and factor2_state < self.factor2_dimension)
        assertion(0 <= multiplicity_index and multiplicity_index < self.multiplicity)
        assertion(0 <= irrep_state and irrep_state < self.irrep_dimension)

        coefficient_label = (factor1_state, factor2_state, multiplicity_index, irrep_state)
        self.clzx[coefficient_label] = value

    # unused
    def highest_weight_normal_form(self):
        hws = self.irrep_dimension - 1

        # bring CGCs into reduced row echelon form
        h = 0
        for i in range(self.factor1_dimension):
            if h<self.multiplicity-1 :
                break
            for j in range(self.factor2_dimension):
                if h<self.multiplicity-1 :
                    break
                k0 = h

                for k in range(h+1,self.multiplicity):
                    if np.abs(self.get(i,j,k,hws)) > np.abs(self.get(i,j,k0,hws)):
                        k0 = k

                if self.get(i,j,k0,hws) < -EPS :
                    for i2 in range(i,self.factor1_dimension):
                        for j2 in range(j if i2==i else 0,self.factor2_dimension):
                            self.set(i2,j2,k0,hws,-self.get(i2,j2,k0,hws))
                elif self.get(i,j,k0,hws) < EPS :
                    continue

                if k0!=h :
                    for i2 in range(i,self.factor1_dimension):
                        for j2 in range(j if i2==i else 0,self.factor2_dimension):
                            x = self.get(i2,j2,k0,hws)
                            self.set(i2,j2,k0,self.get(i2,j2,h,hws))
                            self.set(i2,j2,h,x)

                for k in range(h+1,self.multiplicity):
                    for i2 in range(i,self.factor1_dimension):
                        for j2 in range(j if i2==i else 0,self.factor2_dimension):
                            self.set(i2,j2,k,hws,
                                self.get(i2,j2,k,hws)-self.get(i2,j2,h,hws)*self.get(i,j,k,hws)/self.get(i,j,h,hws)
                                )

                # next 3 lines not strictly necessary, might improve numerical stability
                # or so he says: Atis
                for k in range(h+1,self.multiplicity):
                    self.set(i,j,k,hws,0.0)

        # Gram-Schmidt orthonormalization
        for h in range(self.multiplicity):
            for k in range(h):

                overlap = 0.0

                for i in range(self.factor1_dimension):
                    for j in range(self.factor2_dimension):
                        overlap += self.get(i,j,h,hws) * self.get(i,j,k,hws)

                for i in range(self.factor1_dimension):
                    for j in range(self.factor2_dimension):
                        self.set(i,j,h,hws,
                            self.get(i,j,h,hws)-overlap*self.get(i,j,k,hws)
                            )
            norm = 0.0
            for i in range(self.factor1_dimension):
                for j in range(self.factor2_dimension):
                    norm += self.get(i,j,h,hws)*self.get(i,j,h,hws)
            norm = np.sqrt(norm)

            for i in range(self.factor1_dimension):
                for j in range(self.factor2_dimension):
                    self.set(i,j,h,hws, self.get(i,j,h,hws)/norm )

    def compute_highest_weight_coeffs(self):
        
        if self.multiplicity == 0 :
            return 1

        f1d = self.factor1_dimension
        f2d = self.factor2_dimension
        map_coeff  = np.full((f1d, f2d), -1)
        map_states = np.full((f1d, f2d), -1)

        n_coeff = 0
        n_states = 0
        p = Pattern(self.factor1,0)
        for i in range(f1d):
            pw = Weight(p.get_weight())
            q = Pattern(self.factor2,0)
            for j in range(f2d):
                if pw+q.get_weight() == self.irrep :
                    map_coeff[i,j] = n_coeff
                    n_coeff += 1
                q+=1
            p+=1

        if n_coeff == 1 :
            for i in range(f1d):
                for j in range(f2d):
                    if map_coeff[i,j]>=0 :
                        self.set(i,j,0,self.irrep_dimension-1,1.0)
                        return 1

        hw_system = np.zeros(n_coeff * (f1d*f2d), dtype=np.float64)

        r = Pattern(self.factor1,0)
        for i in range(f1d):
            q = Pattern(self.factor2,0)
            for j in range(f2d):
                if map_coeff[i,j] >= 0 :
                    for l in range(1,self.N) :
                        for k in range(1,l+1) :
                            if (k==1 or r[k,l]+1<=r[k-1,l-1]) and r[k,l]+1 <= r[k,l+1] :
                                r[k,l]+=1
                                h = r.index
                                r[k,l]-=1

                                if map_states[h,j]<0 :
                                    map_states[h,j] = n_states
                                    n_states += 1

                                hw_system[n_coeff * map_states[h,j] + map_coeff[i,j]] += r.raising_coeff(k,l)

                            if (k==1 or q[k,l]+1<=q[k-1,l-1]) and q[k,l]+1 <= q[k,l+1] :
                                q[k,l]+=1
                                h = q.index
                                q[k,l]-=1

                                if map_states[i,h]<0 :
                                    map_states[i,h] = n_states
                                    n_states += 1

                                hw_system[n_coeff * map_states[i,h] + map_coeff[i,j]] += q.raising_coeff(k,l)

                q+=1
            r+=1

        # reshape and truncate
        temp = np.zeros((n_coeff,n_states))
        for i in range(n_coeff) :
            for j in range(n_states) :
                if np.abs(hw_system[i+n_coeff*j])>EPS :
                    temp[i,j] = hw_system[i+n_coeff*j]
        hw_system = temp

        singvec, _, V = svd(hw_system, full_matrices=True)

        # Process the singular vectors
        for i in range(self.multiplicity):
            for j in range(f1d):
                for k in range(f2d):
                    if map_coeff[j,k] >= 0:
                        x = singvec[map_coeff[j,k],n_coeff - 1 - i]
                        
                        if np.abs(x) > EPS:
                            self.set(j, k, i, self.irrep_dimension - 1, x)

        #self.highest_weight_normal_form()

    def compute_lower_weight_coeffs(self, multip_index, state, done):

        statew = Weight(Pattern(self.irrep,state).get_weight())
        p = Pattern(self.irrep,0)
        map_parent = [-1]*self.irrep_dimension
        map_multi  = [-1]*self.irrep_dimension
        which_l    = [-1]*self.irrep_dimension
        n_parent = 0
        n_multi  = 0

        for i in range(self.irrep_dimension):
            v = Weight(p.get_weight())

            if v==statew:
                map_multi[i] = n_multi
                n_multi += 1
            else:
                for l in range(1,self.N):
                    v[l]-=1
                    v[l+1]+=1
                    if v==statew :
                        map_parent[i] = n_parent
                        n_parent+=1
                        which_l[i] = l
                        if not done[i] :
                            compute_lower_weight_coeffs(multip_index, i, done)
                        break
                    v[l+1]-=1
                    v[l]+=1
            p+=1

        f1d = self.factor1_dimension
        f2d = self.factor2_dimension
        irrep_coeffs = np.zeros(n_parent * n_multi, dtype=np.float64)
        prod_coeffs  = np.zeros(n_parent * (f1d*f2d), dtype=np.float64)

        map_prodstat = np.full((f1d, f2d), -1)

        n_prodstat = 0
        r = Pattern(self.irrep,0)
        for i in range(self.irrep_dimension):
            if map_parent[i]>=0 :
                l = which_l[i]
                for k in range(1,l+1):
                    if r[k,l]>r[k+1,l+1] and (k==l or r[k,l]>r[k,l-1]) :
                        r[k,l]-=1
                        h = r.index
                        r[k,l]+=1
                        irrep_coeffs[n_parent * map_multi[h] + map_parent[i]] += r.lowering_coeff(k, l)

                q1 = Pattern(self.factor1,0)
                for j1 in range(f1d):
                    q2 = Pattern(self.factor2,0)
                    for j2 in range(f2d):
                        
                        if np.abs(self.get(j1,j2,multip_index,i))>EPS:
                            l = which_l[i]
                            for k in range(1,l+1):
                                
                                if q1[k,l]>q1[k+1,l+1] and (k==l or q1[k,l]>q1[k,l-1]):
                                    q1[k,l]-=1
                                    h = q1.index
                                    q1[k,l]+=1

                                    if map_prodstat[h,j2]<0 :
                                        map_prodstat[h,j2] = n_prodstat
                                        n_prodstat+=1

                                    prod_coeffs[n_parent * map_prodstat[h,j2] + map_parent[i]] +=(
                                        self.get(j1,j2,multip_index,i)*q1.lowering_coeff(k, l)
                                        )


                                if q2[k,l]>q2[k+1,l+1] and (k==l or q2[k,l]>q2[k,l-1]):
                                    q2[k,l]-=1
                                    h = q2.index
                                    q2[k,l]+=1

                                    if map_prodstat[j1,h]<0 :
                                        map_prodstat[j1,h] = n_prodstat
                                        n_prodstat+=1

                                    prod_coeffs[n_parent * map_prodstat[j1,h] + map_parent[i]] +=(
                                        self.get(j1,j2,multip_index,i)*q2.lowering_coeff(k, l)
                                        )

                        q2+=1
                    q1+=1
            r+=1

        # doing the least square method

        #reshape and truncate
        
        temp1 = np.zeros((n_parent,n_multi))
        for i in range(n_parent) :
            for j in range(n_multi) :
                if np.abs(irrep_coeffs[i+n_parent*j])>EPS :
                    temp1[i,j] = irrep_coeffs[i+n_parent*j]
        irrep_coeffs = temp1

        #reshape and truncate
        temp2 = np.zeros((n_parent,n_prodstat))
        for i in range(n_parent) :
            for j in range(n_prodstat) :
                if np.abs(prod_coeffs[i+n_parent*j])>EPS :
                    temp2[i,j] = prod_coeffs[i+n_parent*j]
        prod_coeffs = temp2
        
        prod_coeffs, _, _, _ = np.linalg.lstsq(irrep_coeffs, prod_coeffs, rcond=None)


        for i in range(self.irrep_dimension):
            if map_multi[i] >= 0:
                for j in range(f1d):
                    for k in range(f2d):
                        if map_prodstat[j,k] >= 0:
                            x = prod_coeffs[map_multi[i], map_prodstat[j,k]]

                            if np.abs(x) > EPS:
                                self.set(j, k, multip_index, i, x)

                done[i] = True

# ==== Utilities ==================================================================================

def assertion(condition,message="Assertion condition is not met"):
    if not condition :
        raise SystemExit(message)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Above here are the computations. Do not touch unless you know what you are doing.
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def ordinal(n):
    if str(n)[-1]=="1" and n!=11:
        return str(n)+"st"
    elif str(n)[-1]=="2" and n!=12:
        return str(n)+"nd"
    elif str(n)[-1]=="3" and n!=13:
        return str(n)+"rd"
    else:
        return str(n)+"th"

# for computing the CG multiplication
def contract_CGs(coeff_lists):
    """
    Compute the contracted tensor A[i1, i2, ..., in, a] from a list of coefficient lists.
    
    Args:
        coeff_lists: List of coefficient lists [C1, C2, ..., C_{n-1}], where each Cm
                     is a list of tuples (i, j, k, val) representing nonzero elements
                     of a 3D sparse tensor. For m=1 to n-1, Cm has indices interpreted as:
                     - C1: (i1, i2, k1, val)
                     - Cm (2 ≤ m ≤ n-2): (k_{m-1}, i_{m+1}, k_m, val)
                     - C_{n-1}: (k_{n-2}, in, a, val)
                     The result A has n+1 indices (i1, i2, ..., in, a).
    
    Returns:
        List of tuples (i1, i2, ..., in, a, val) representing nonzero elements of A.
    """
    # Handle the base case where n=2, so only C1 exists and A = C1
    if len(coeff_lists) == 1:
        return coeff_lists[0]
    
    # Initialize the intermediate tensor T by contracting C1 and C2
    C1 = coeff_lists[0]  # List of (i1, i2, k1, val)
    C2 = coeff_lists[1]  # List of (k1, i3, k2, val)
    
    # Group C2 by k1 (first index) for efficient contraction
    C2_group = defaultdict(list)
    for (i, j, k, val) in C2:
        C2_group[i].append((j, k, val))  # Store (i3, k2, val) for each k1
    
    # Compute T[i1, i2, i3, k2] = sum_{k1} C1[i1, i2, k1] * C2[k1, i3, k2]
    T = defaultdict(float)
    for (i1, i2, k1, val1) in C1:
        for (i3, k2, val2) in C2_group[k1]:
            key = (i1, i2, i3, k2)
            T[key] += val1 * val2
    
    # Sequentially contract T with C3, C4, ..., C_{n-1}
    for m in range(2, len(coeff_lists)):
        Cm = coeff_lists[m]  # List of (k_{m-1}, i_{m+1}, k_m, val), or (k_{n-2}, in, a, val) if m=n-1
        
        # Group Cm by k_{m-1} (first index)
        Cm_group = defaultdict(list)
        for (i, j, k, val) in Cm:
            Cm_group[i].append((j, k, val))  # Store (i_{m+1}, k_m, val) or (in, a, val)
        
        # Contract T with Cm over k_{m-1}
        T_new = defaultdict(float)
        for key, val_T in T.items():
            k_m_prev = key[-1]  # Last index of T is k_{m-1}
            for (j, k, val_C) in Cm_group[k_m_prev]:
                # New key is all indices of T except the last, plus Cm's j and k
                new_key = key[:-1] + (j, k)
                T_new[new_key] += val_T * val_C
        T = T_new
    
    # Convert the final tensor T to a list of tuples (i1, i2, ..., in, a, val)
    # Filter out zero values (though typically rare with sparse inputs)
    result = [(key + (val,)) for key, val in T.items() if val != 0]
    
    return result

def parse_einsum(einsum_str):
    """
    Parse an einsum string into per-operand subscripts and an output subscript.
    For example, "ijk,klm->iljm" returns (["ijk", "klm"], "iljm").
    """
    inputs, output = einsum_str.split("->")
    subscript_list = inputs.split(",")
    return subscript_list, output

def contract_inner(inner1, inner2, einsum_str):
    """
    Contract two inner tensors using an einsum-like notation.

    inner1, inner2: lists of tuples representing nonzero components.
       Each tuple is (i1, i2, ..., value) where the first N elements are the indices,
       and the last element is the numeric value.
    einsum_str: a string like "ijk,klm->iljm"

    Returns: an inner tensor in the same format (a list of tuples)
    """
    subscripts, out_sub = parse_einsum(einsum_str)
    sub1, sub2 = subscripts

    # Identify contracted indices: letters that appear in both input subscripts but not in output.
    contracted_letters = set(sub1) & set(sub2) - set(out_sub)

    # Build dictionaries mapping each letter to its position in the respective operand.
    pos1 = {letter: pos for pos, letter in enumerate(sub1)}
    pos2 = {letter: pos for pos, letter in enumerate(sub2)}

    # For the output, for each letter in out_sub, decide which operand it comes from.
    out_map = []
    for letter in out_sub:
        if letter in pos1:
            out_map.append(('op1', pos1[letter]))
        elif letter in pos2:
            out_map.append(('op2', pos2[letter]))
        else:
            raise ValueError(f"Output index letter '{letter}' not present in any operand.")

    # Use a dictionary to collect the summed result.
    result = defaultdict(float)

    # Loop over nonzero components from inner1 and inner2.
    for comp1 in inner1:
        indices1 = comp1[:-1]
        val1 = comp1[-1]
        for comp2 in inner2:
            indices2 = comp2[:-1]
            val2 = comp2[-1]

            # Check if contracted indices match.
            valid = True
            for letter in contracted_letters:
                if indices1[pos1[letter]] != indices2[pos2[letter]]:
                    valid = False
                    break
            if not valid:
                continue

            # Build the output index tuple.
            out_indices = []
            for source, pos in out_map:
                if source == 'op1':
                    out_indices.append(indices1[pos])
                else:
                    out_indices.append(indices2[pos])
            result[tuple(out_indices)] += val1 * val2

    # Return result in the same list-of-tuples format.
    return [(*indices, value) for indices, value in result.items()]

def contract_outer(outer1, outer2, einsum_str_outer, einsum_str_inner):
    """
    Contract two outer tensors.

    outer1, outer2: lists of blocks.
       Each block is a tuple: (outer indices..., inner_tensor)
       where inner_tensor is in the same format as used for inner tensors.
       
    einsum_str_outer: a string in einsum notation for outer indices.
       For example, "abc,acd->bd" means: match outer indices 'a' and 'c' from the two tensors 
       (contracting over them) and produce a result with free outer indices (here 'b' and 'd').
    einsum_str_inner: a string in einsum notation for inner contraction,
       e.g. "ijk,klm->iljm"
       
    Returns: an outer tensor in the same format (a list of blocks),
       where each block is (output outer indices..., inner_tensor) and inner_tensor is a list of tuples.
    """
    # Parse the outer einsum.
    outer_subs, outer_out = parse_einsum(einsum_str_outer)
    sub1_outer, sub2_outer = outer_subs

    pos1_outer = {letter: pos for pos, letter in enumerate(sub1_outer)}
    pos2_outer = {letter: pos for pos, letter in enumerate(sub2_outer)}

    # Contracted outer indices: those in both inputs but not in output.
    contracted_outer = set(sub1_outer) & set(sub2_outer) - set(outer_out)

    # Build output mapping for outer indices.
    out_map_outer = []
    for letter in outer_out:
        if letter in pos1_outer:
            out_map_outer.append(('op1', pos1_outer[letter]))
        elif letter in pos2_outer:
            out_map_outer.append(('op2', pos2_outer[letter]))
        else:
            raise ValueError(f"Output outer index letter '{letter}' not present in any operand.")

    # For efficiency, put outer blocks into dictionaries.
    # Assume each block is a tuple: (outer_index0, outer_index1, ..., inner_tensor)
    outer1_dict = {}
    for block in outer1:
        outer_idx = block[:-1]
        inner_tensor = block[-1]
        outer1_dict[outer_idx] = inner_tensor

    outer2_dict = {}
    for block in outer2:
        outer_idx = block[:-1]
        inner_tensor = block[-1]
        outer2_dict[outer_idx] = inner_tensor

    # Dictionary to accumulate results: maps output outer indices tuple -> inner tensor (accumulated as a dict)
    outer_result = defaultdict(lambda: defaultdict(float))

    # Loop over all pairs of outer blocks.
    for outer_idx1, inner_tensor1 in outer1_dict.items():
        for outer_idx2, inner_tensor2 in outer2_dict.items():
            # Check that contracted outer indices match.
            valid = True
            for letter in contracted_outer:
                if outer_idx1[pos1_outer[letter]] != outer_idx2[pos2_outer[letter]]:
                    valid = False
                    break
            if not valid:
                continue

            # Build the output outer indices tuple.
            out_outer = []
            for source, pos in out_map_outer:
                if source == 'op1':
                    out_outer.append(outer_idx1[pos])
                else:
                    out_outer.append(outer_idx2[pos])
            out_outer = tuple(out_outer)

            # Contract the inner tensors using the provided inner einsum.
            contracted_inner = contract_inner(inner_tensor1, inner_tensor2, einsum_str_inner)
            # Add the contracted inner components to the result.
            for comp in contracted_inner:
                inner_indices = comp[:-1]
                val = comp[-1]
                outer_result[out_outer][inner_indices] += val

    # Convert the inner dictionaries into lists of tuples.
    final_outer = []
    for out_outer, inner_dict in outer_result.items():
        inner_list = [(*indices, value) for indices, value in inner_dict.items()]
        # Each block is (outer indices..., inner_tensor)
        final_outer.append((*out_outer, inner_list))
    return final_outer

class SparseCoeff:
	
	def __init__(self, coeffs, shape, indices):
		self.coeffs = coeffs
		self.shape = tuple([*shape])
		self.indices = indices

	@property
	def nnz(self):
		return len(self.coeffs)

	@property
	def size(self):
		return np.prod(self.shape)

# ==== Main setup =================================================================================

use_N_weight_notation = False

irrep_set = []
irrep_set.append( [ [0,0], [1,0], [1,1], [2,1] ] ) # irrep set for term 0
irrep_set.append( [ [0,0], [1,0], [1,1] ] ) # irrep set for term 1


axis_info = {}
axis_info["U0"] = [0, 0, 0, 1] # a list of terms where each U0 came from
axis_info["V0"] = [0, 0, 0]
axis_info["U1"] = [0, 0, 0]
axis_info["V1"] = [0, 0, 0]
axis_info["U2"] = [0, 0, 0]
axis_info["V2"] = [0, 0, 0]
axis_info["U3"] = [0, 0, 0]
axis_info["V3"] = [0, 0, 0]


connection_info = {}
connection_info["-U1 to +U0"] = [0] # a list of terms where
connection_info["-U0 to -V1"] = [0] # these connections came from.
connection_info["+V1 to -V0"] = [0]
connection_info["+V0 to +U1"] = [0]
connection_info["-U2 to +U0"] = [0]
connection_info["-U0 to -V2"] = [0]
connection_info["+V2 to -V0"] = [0]
connection_info["+V0 to +U2"] = [0]
connection_info["-U2 to +U1"] = [0]
connection_info["-U1 to -V2"] = [0]
connection_info["+V2 to -V1"] = [0]
connection_info["+V1 to +U2"] = [0]
connection_info["-U3 to +U0"] = [0]
connection_info["-U0 to -V3"] = [0]
connection_info["+V3 to -V0"] = [0]
connection_info["+V0 to +U3"] = [0]
connection_info["-U3 to +U1"] = [0]
connection_info["-U1 to -V3"] = [0]
connection_info["+V3 to -V1"] = [0]
connection_info["+V1 to +U3"] = [0]
connection_info["-U3 to +U2"] = [0]
connection_info["-U2 to -V3"] = [0]
connection_info["+V3 to -V2"] = [0]
connection_info["+V2 to +U3"] = [0]
connection_info["-U0 to +U0"] = [1]

main()