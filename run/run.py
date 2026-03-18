import numpy as np
from dedalus import public as de
import time as time_module
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_min_safe_Nr(Ntheta, safety_margin=4):
    """Calculates minimum radial resolution to avoid array size errors in BallBasis."""
    return int(np.ceil(Ntheta / 2)) + safety_margin

def save_diagnostic_pair(data, r_grid, theta_grid, B_lim, err_min, err_max, prefix):
    """Generates high-fidelity snapshots of the toroidal field and absolute error."""
    count, sim_time, ratio = data['count'], data['sim_time'], data['ratio']
    B_phi, A_phi, err = data['B_phi'], data['A_phi'], data['err']
    
    r, th = r_grid[0, :, :], theta_grid[0, :, :]
    Psi = r * np.sin(th) * A_phi
    x, z = r * np.sin(th), r * np.cos(th)
    
    # Toroidal Field Visualization
    plt.figure(figsize=(6, 8))
    cf = plt.pcolormesh(x, z, B_phi, shading='gouraud', cmap='RdBu_r', vmin=-B_lim, vmax=B_lim)
    plt.colorbar(cf, label=r'$B_\phi$')
    plt.contour(x, z, Psi, colors='k', linewidths=0.8, levels=15, alpha=0.5)
    
    plt.title(f"t = {sim_time:.4f} | Ratio: {ratio:.2f}")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{prefix}_snap_{count:02d}.png")
    plt.close()

    # Absolute Error Mapping
    plt.figure(figsize=(6, 8))
    norm_err = colors.LogNorm(vmin=err_min, vmax=err_max)
    cf2 = plt.pcolormesh(x, z, err + 1e-18, cmap='inferno', norm=norm_err)
    plt.colorbar(cf2, label='Absolute Error')
    
    plt.title(f"Error Map | t = {sim_time:.4f}")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{prefix}_error_{count:02d}.png")
    plt.close()

def run_simulation(Nphi=4, Ntheta=32, Nr=32, make_plots=False):
    """Solver for magnetic induction in a spherical ball (r=0 included) using spectral methods."""
    min_Nr = get_min_safe_Nr(Ntheta)
    if Nr < min_Nr: 
        Nr = min_Nr
    
    R_star, target_ratio = 1.0, 5.0
    omega_c, alpha = 60.0, 180.0
    
    coords = de.SphericalCoordinates('phi', 'theta', 'r')
    dist = de.Distributor(coords, dtype=np.float64)
    basis = de.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=R_star, dtype=np.float64)

    A = dist.VectorField(coords, name='A', bases=basis)
    v = dist.VectorField(coords, name='v', bases=basis)
    phi_g = dist.Field(name='phi_g', bases=basis)
    tau_A = dist.VectorField(coords, name='tau_A', bases=basis.surface)
    tau_phi = dist.Field(name='tau_phi', bases=basis.surface)

    problem = de.IVP([A, phi_g, tau_A, tau_phi], namespace={
        'v': v, 'curl': de.curl, 'cross': de.cross, 'grad': de.grad, 
        'div': de.div, 'lap': de.lap, 'lift': lambda f: de.Lift(f, basis, -1)
    })
    
    problem.add_equation("dt(A) + grad(phi_g) + lift(tau_A) = cross(v, curl(A))")
    problem.add_equation("lap(phi_g) + lift(tau_phi) = div(cross(v, curl(A)))")
    problem.add_equation("A(r=1.0) = 0")
    problem.add_equation("phi_g(r=1.0) = 0")

    solver = problem.build_solver(de.RK443)
    phi_grid, theta_grid, r_grid = dist.local_grids(basis, scales=1)
    
    v['g'][0] = (omega_c + alpha * r_grid) * r_grid * np.sin(theta_grid)
    A['g'][0] = 1.0 * (r_grid**2) * np.sin(theta_grid)

    cfl = de.CFL(solver, initial_dt=1e-5, cadence=10, safety=0.5, max_dt=1e-3)
    cfl.add_velocity(v)

    start_time = time_module.time()
    img_count, next_threshold = 0, 0.0
    plot_data = []
    
    while solver.proceed:
        dt = cfl.compute_timestep()
        solver.step(dt)
        
        if solver.iteration % 20 == 0:
            B_f = de.curl(A).evaluate()
            ratio = np.sqrt(np.mean(B_f['g'][0]**2) / (np.mean(B_f['g'][1]**2 + B_f['g'][2]**2) + 1e-16))
            
            if make_plots and ratio >= next_threshold and img_count < 10:
                B_phi_data = B_f['g'][0][0, :, :].copy()
                A_phi_data = A['g'][0][0, :, :].copy()
                B_phi_ana = (2 * alpha * (r_grid[0,:,:]**2) * np.sin(theta_grid[0,:,:]) * np.cos(theta_grid[0,:,:]) * solver.sim_time)
                
                plot_data.append({
                    'count': img_count, 'sim_time': solver.sim_time, 'ratio': ratio,
                    'B_phi': B_phi_data, 'A_phi': A_phi_data, 'err': np.abs(B_phi_data - B_phi_ana)
                })
                img_count += 1
                next_threshold += (target_ratio / 10.0) 
            
            if ratio >= target_ratio: 
                break

    exec_time = time_module.time() - start_time
    
    if make_plots and plot_data:
        all_B = np.array([d['B_phi'] for d in plot_data])
        all_err = np.array([d['err'] for d in plot_data])
        
        B_lim = max(np.abs(np.max(all_B)), np.abs(np.min(all_B)))
        if B_lim == 0: 
            B_lim = 1.0
        
        valid_errs = all_err[all_err > 1e-15]
        err_min = np.min(valid_errs) if len(valid_errs) > 0 else 1e-12
        err_max = np.max(all_err) if len(all_err) > 0 else err_min * 10.0
        if err_max <= err_min: 
            err_max = err_min * 10.0
        
        for d in plot_data:
            save_diagnostic_pair(d, r_grid, theta_grid, B_lim, err_min, err_max, prefix="comp")

    A.change_scales(1)
    B_final = de.curl(A).evaluate()
    B_final.change_scales(1)
    B_energy = np.sqrt(np.mean(B_final['g']**2))

    return exec_time, B_energy

def run_comparison(param_name, param_list, base_Ntheta=32):
    """Performs resolution sweeps to verify spectral convergence."""
    logger.info(f"Convergence Test: {param_name}")
    times = []
    energy_dict = {}
    
    for val in param_list:
        configs = {'Nphi': 4, 'Ntheta': base_Ntheta, 'Nr': 64}
        configs[param_name] = val
        t, b_en = run_simulation(**configs)
        times.append(t)
        energy_dict[val] = b_en
        logger.info(f"{param_name}={val} | Time: {t:.2f}s | B_Energy: {b_en:.6e}")

    ref_val = param_list[-1]
    ref_energy = energy_dict[ref_val]
    rel_errors = [np.abs(energy_dict[v] - ref_energy) / np.abs(ref_energy) + 1e-16 for v in param_list]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(param_list, rel_errors, 'rs-', linewidth=2, label='Relative Error')
    ax1.set_yscale('log')
    ax1.set_ylabel('Relative Error', color='r')
    ax1.set_xlabel(f'Resolution ({param_name})')
    
    ax2 = ax1.twinx()
    ax2.plot(param_list, times, 'bo--', label='Execution Time')
    ax2.set_ylabel('Execution Time (s)', color='b')
    
    plt.title(f'Spectral Convergence: {param_name}')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'comp_convergence_{param_name}.png')
    plt.close()

if __name__ == "__main__":
    logger.info("Starting Primary Simulation Run...")
    run_simulation(Nphi=4, Ntheta=64, Nr=64, make_plots=True)
    
    logger.info("Starting Convergence Analysis...")
    run_comparison('Nphi', [4, 8, 16, 32, 64], base_Ntheta=16)
    run_comparison('Nr', [16, 32, 48, 64, 80], base_Ntheta=16)
    run_comparison('Ntheta', [16, 32, 48, 64], base_Ntheta=32)
