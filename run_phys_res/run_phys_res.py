import numpy as np
from dedalus import public as de
import time as time_module
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_diagnostic_pair(data, r_grid, theta_grid, B_lim, err_max, prefix):
    """Generates high-fidelity snapshots of the toroidal field and absolute error."""
    count, sim_time, ratio = data['count'], data['sim_time'], data['ratio']
    B_phi, Psi, err_abs = data['B_phi'], data['Psi'], data['err_abs']
    
    x = r_grid[0,:,:] * np.sin(theta_grid[0,:,:])
    z = r_grid[0,:,:] * np.cos(theta_grid[0,:,:])
    
    # Toroidal Field Visualization
    plt.figure(figsize=(6, 8))
    norm = colors.SymLogNorm(linthresh=1e-6, vmin=-B_lim, vmax=B_lim, base=10)
    cf = plt.pcolormesh(x, z, B_phi, shading='gouraud', cmap='RdBu_r', norm=norm)
    plt.colorbar(cf, label=r'$B_\phi$')
    
    if np.max(np.abs(Psi)) > 1e-12:
        plt.contour(x, z, Psi, colors='k', linewidths=0.5, levels=14, alpha=0.3)
    
    plt.title(f"t = {sim_time:.3f} | Ratio: {ratio:.2f}")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{prefix}_snap_{count:02d}.png")
    plt.close()

    # Error Visualization
    plt.figure(figsize=(6, 8))
    err_pct = (err_abs / B_lim) * 100
    norm_err = colors.LogNorm(vmin=1e-10, vmax=max(err_max, 1.0))
    cf2 = plt.pcolormesh(x, z, err_pct, cmap='inferno', norm=norm_err) 
    plt.colorbar(cf2, label='Error (% of Final Peak B)')
    
    plt.title(f"Error Map | t = {sim_time:.4f}")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{prefix}_error_{count:02d}.png")
    plt.close()

def run_simulation(Nphi=4, Ntheta=64, Nr=96, make_plots=False):
    """Solver for magnetic induction in a spherical shell with Ohmic dissipation."""
    R_in, R_out = 0.2, 1.0
    omega_c, alpha = 60.0, 40.0
    eta = 5e-3 
    
    coords = de.SphericalCoordinates('phi', 'theta', 'r')
    dist = de.Distributor(coords, dtype=np.float64)
    basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(R_in, R_out), dtype=np.float64)

    A = dist.VectorField(coords, name='A', bases=basis)
    v = dist.VectorField(coords, name='v', bases=basis)
    phi_g = dist.Field(name='phi_g', bases=basis)
    
    tau_A1 = dist.VectorField(coords, name='tau_A1', bases=basis.S2_basis())
    tau_A2 = dist.VectorField(coords, name='tau_A2', bases=basis.S2_basis())
    tau_phi1 = dist.Field(name='tau_phi1', bases=basis.S2_basis())
    tau_phi2 = dist.Field(name='tau_phi2', bases=basis.S2_basis())

    problem = de.IVP([A, phi_g, tau_A1, tau_A2, tau_phi1, tau_phi2], namespace={
        'v': v, 'curl': de.curl, 'grad': de.grad, 'div': de.div, 
        'lap': de.lap, 'lift': lambda f, n: de.Lift(f, basis, n), 'cross': de.cross,
        'eta': eta
    })
    
    phi_grid, theta_grid, r_grid = dist.local_grids(basis, scales=1)
    
    A_ic = dist.VectorField(coords, name='A_ic', bases=basis)
    A_ic['g'][0] = 1.0 * np.sin(theta_grid) / (r_grid**2)
    A['g'] = A_ic['g'].copy()
    
    v['g'][0] = 0.0 

    problem.namespace['A_ic'] = A_ic
    problem.add_equation("dt(A) - eta * lap(A) + grad(phi_g) + lift(tau_A1, -1) + lift(tau_A2, -2) = cross(v, curl(A))")
    problem.add_equation("lap(phi_g) + lift(tau_phi1, -1) + lift(tau_phi2, -2) = div(cross(v, curl(A)))")
    problem.add_equation("A(r=1.0) = A_ic(r=1.0)")
    problem.add_equation(f"A(r={R_in}) = A_ic(r={R_in})")
    problem.add_equation("phi_g(r=1.0) = 0")
    problem.add_equation(f"phi_g(r={R_in}) = 0")

    solver = problem.build_solver(de.RK443)
    cfl = de.CFL(solver, initial_dt=1e-5, cadence=10, safety=0.4, max_dt=5e-4)
    cfl.add_velocity(v)

    start_time = time_module.time()
    
    img_count = 0
    plot_interval = 0.1
    next_plot_time = 0.0
    plot_data = []
    
    # Steady state tracking
    prev_B_tor = 0.0
    steady_state_tolerance = 5e-3
    min_sim_time = 0.1

    logger.info(f"Starting Shell Simulation: Nphi={Nphi}, Ntheta={Ntheta}, Nr={Nr}")

    while solver.proceed:
        # Velocity Ramping for stable ICs
        ramp = min(1.0, solver.sim_time / 0.05)
        v['g'][0] = ramp * (omega_c + alpha * r_grid) * r_grid * np.sin(theta_grid)
        
        dt = cfl.compute_timestep()
        solver.step(dt)
        
        if solver.iteration % 10 == 0:
            B_f = de.curl(A).evaluate()
            B_tor_rms = np.sqrt(np.mean(B_f['g'][0]**2))
            B_pol_rms = np.sqrt(np.mean(B_f['g'][1]**2 + B_f['g'][2]**2))
            ratio_val = B_tor_rms / (B_pol_rms + 1e-16)
            
            if solver.iteration % 50 == 0:
                rel_change = np.abs(B_tor_rms - prev_B_tor) / (B_tor_rms + 1e-16)
                logger.info(f"Iter: {solver.iteration:05d} | Time: {solver.sim_time:.4f} | dt: {dt:.2e} | Ratio: {ratio_val:.4f} | Rel Change: {rel_change:.2e}")
            
            if make_plots and solver.sim_time >= next_plot_time and img_count < 10:
                B_phi_data = B_f['g'][0][0, :, :].copy()
                B_phi_ana = (2 * alpha * np.sin(theta_grid[0,:,:]) * np.cos(theta_grid[0,:,:]) * solver.sim_time) / (r_grid[0,:,:]**2)
                
                plot_data.append({
                    'count': img_count, 'sim_time': solver.sim_time, 'ratio': ratio_val,
                    'B_phi': B_phi_data, 'Psi': (r_grid * np.sin(theta_grid) * A['g'][0])[0,:,:].copy(),
                    'err_abs': np.abs(B_phi_data - B_phi_ana)
                })
                img_count += 1
                next_plot_time += plot_interval
            
            # Equilibrium Checking
            if solver.sim_time > min_sim_time:
                rel_change = np.abs(B_tor_rms - prev_B_tor) / (B_tor_rms + 1e-16)
                if rel_change < steady_state_tolerance:
                    logger.info(f"Equilibrium reached at t={solver.sim_time:.4f} (rel_change={rel_change:.2e}). Stopping.")
                    
                    if make_plots and img_count < 10:
                        B_phi_data = B_f['g'][0][0, :, :].copy()
                        plot_data.append({
                            'count': img_count, 'sim_time': solver.sim_time, 'ratio': ratio_val,
                            'B_phi': B_phi_data, 'Psi': (r_grid * np.sin(theta_grid) * A['g'][0])[0,:,:].copy(),
                            'err_abs': np.abs(B_phi_data - (2 * alpha * np.sin(theta_grid[0,:,:]) * np.cos(theta_grid[0,:,:]) * solver.sim_time) / (r_grid[0,:,:]**2))
                        })
                    break
                    
            prev_B_tor = B_tor_rms

    exec_time = time_module.time() - start_time

    if make_plots and plot_data:
        B_lim = max([np.max(np.abs(d['B_phi'])) for d in plot_data])
        err_max = max([np.max((d['err_abs'] / B_lim) * 100) for d in plot_data])
        for d in plot_data:
            save_diagnostic_pair(d, r_grid, theta_grid, B_lim, err_max, prefix="phys")

    A.change_scales(1)
    B_final = de.curl(A).evaluate()
    B_final.change_scales(1)
    B_energy = np.sqrt(np.mean(B_final['g']**2))
    
    return exec_time, B_energy

def run_comparison(param_name, param_list, base_Ntheta=32, base_Nr=64):
    """Performs resolution sweeps to verify spectral convergence for the ShellBasis."""
    logger.info(f"Convergence Test: {param_name}")
    times = []
    energy_dict = {}
    
    for val in param_list:
        configs = {'Nphi': 4, 'Ntheta': base_Ntheta, 'Nr': base_Nr}
        configs[param_name] = val
        t, b_en = run_simulation(**configs)
        times.append(t)
        energy_dict[val] = b_en
        logger.info(f"{param_name}={val} | Time: {t:.2f}s | B_Energy: {b_en:.6e}")

    ref_val = param_list[-1]
    ref_energy = energy_dict[ref_val]
    rel_errors = [np.abs(energy_dict[v] - ref_energy) / (np.abs(ref_energy) + 1e-16) + 1e-16 for v in param_list]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(param_list, rel_errors, 'rs-', linewidth=2, label='Relative Error')
    ax1.set_yscale('log')
    ax1.set_ylabel('Relative Energy Error', color='r')
    ax1.set_xlabel(f'Resolution ({param_name})')
    
    ax2 = ax1.twinx()
    ax2.plot(param_list, times, 'bo--', label='Execution Time')
    ax2.set_ylabel('Execution Time (s)', color='b')
    
    plt.title(f'Spectral Convergence: {param_name} (Shell Model)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'phys_convergence_{param_name}.png')
    plt.close()

if __name__ == "__main__":
    # Primary Simulation Run
    logger.info("Starting Primary Shell Simulation Run...")
    run_simulation(Nphi=4, Ntheta=64, Nr=96, make_plots=True)
    
    # Convergence Analysis
    logger.info("Starting Convergence Analysis for Shell Model...")
    run_comparison('Nphi', [4, 8, 16, 32], base_Ntheta=16, base_Nr=48)
    run_comparison('Ntheta', [16, 32, 48, 64], base_Ntheta=32, base_Nr=64)
    run_comparison('Nr', [32, 48, 64, 80, 96], base_Ntheta=32, base_Nr=64)
