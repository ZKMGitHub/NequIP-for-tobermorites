# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# we must undefine any fix ave/* fix before using reset_timestep
if "$(is_defined(fix,avp))" then "unfix avp"
reset_timestep 0


# Choose potential
# pair_style	sw
# pair_coeff * * Si.sw Si

pair_style	deepmd tob11_deepmd_compress.pb
pair_coeff	* * Ca H O Si 

atom_modify sort 0 0.0    
#kspace_style	ewald 1.0e-6# Setup neighbor style

#neighbor 2.0 bin
neigh_modify once no every 1 delay 0 check yes



# Setup output

fix avp all ave/time  ${nevery} ${nrepeat} ${nfreq} c_thermo_press mode vector

#min_style cg
#minimize 1.0e-15 1.0e-15 10000 10000 #设置能量最小化

thermo		${nthermo}
thermo_style custom step temp pe press f_avp[1] f_avp[2] f_avp[3] f_avp[4] f_avp[5] f_avp[6]
thermo_modify norm no

# Setup MD

timestep ${timestep}
fix 4 all nve
if "${thermostat} == 1" then &
   "fix 5 all langevin ${temp} ${temp} ${tdamp} ${seed}"


