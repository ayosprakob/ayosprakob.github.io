python_src = `
'''
  ::: clebsch.py :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  This is the Python script converted from Arne Alex's C++ code provided in
  https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/ClebschGordan.cpp
  This file contains everything in the namespace "clebsch"
               
  Thanks, Arne.

  -- Converted by Atis Yosprakob // 3 Mar 2025

  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
'''


# Argument passed from the webpage -- Modify this to whatever you want

weight_input1 = __weight1__  # List; Nth component is omitted for SU(N)
weight_input2 = __weight2__  # List

weight_input3 = __weight3__  # List; the resulting representation (None means all)
weight_mult3  = __multip3__  # Int; multiplicity index (None means all)


def main():
    w1 = Weight(weight_input1)
    w2 = Weight(weight_input2)
    decomp = Decomposition(w1,w2)

    w3_expected = None if weight_input3 == None else Weight(weight_input3)
    m_expected = weight_mult3

    d1 = w1.dimension
    d2 = w2.dimension
    d3total = w1.dimension*w2.dimension

    jsprint("Group: SU("+str(w1.N)+")")
    jsprint()

    ishift = 0
    for w3 in decomp:
        if w3_expected != None and w3 != w3_expected:
            continue

        mult = decomp.multiplicity(w3)

        d3 = w3.dimension
        for m in range(1,mult+1):
            if m_expected != None and m!=m_expected :
                continue
            jsprint(w1,"⊗",w2,"→",w3,"("+ordinal(m)+" multiplicity)" if mult>1 else "")
            jsprint("Coefficient dimensions:",d1,"×",d2,"×",d3)
            jsprint()
            CGmat = np.zeros((d1,d2,d3))
            CGi = Coefficients(w3,w1,w2,m-1).tensor
            for i1,i2,i,val in CGi:
                #CGmat[i1,i2,i+ishift] = val
                CGmat[i1,i2,i] = val
                jsprint("C"+str([i1,i2,i]),"=",val)
            ishift+=d3

            
            # Orthogonality test
            err = np.linalg.norm(np.einsum('ija,ijb->ab',CGmat,CGmat)-np.identity(d3))
            jsprint()
            jsprint("Orthogonality error:",err)
            jsprint()
            jsprint()
            jsprint()

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Below here are the computations. Do not touch unless you know what you are doing.
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
                result+="\\n"
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
                result+="\\n"+"  + "
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
                result += str(key)+" : "+str(coeff_entries[key])+"\\n"
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

# For printing in web assembly
def jsprint(*args,end="\\n"):
    ret = ""
    for i,elem in enumerate(args):
        ret += str(elem)
        if i!=len(args) :
            ret += " "
        else:
            ret += end
    print(ret)

def assertion(condition,message="Assertion condition is not met"):
    if not condition :
        raise SystemExit(message)
      
def ordinal(n):
    if str(n)[-1]=="1" and n!=11:
        return str(n)+"st"
    elif str(n)[-1]=="2" and n!=12:
        return str(n)+"nd"
    elif str(n)[-1]=="3" and n!=13:
        return str(n)+"rd"
    else:
        return str(n)+"th"
  
use_N_weight_notation = False

main()
        `