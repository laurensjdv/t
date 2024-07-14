import torch
import torch.nn.functional as F
import math

import numpy as np

from tqdm import tqdm
from .LBFGS import *


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()
        ##################
        ### added code ###
        ##################
        self.size = out_features * in_features 
        # self.scale_bases = torch.nn.Parameter(torch.ones(self.size) * scale_base)#.requires_grad_(True)
        # self.scale_sp = torch.nn.Parameter(torch.ones(self.size) * scale_spline)#.requires_grad_(True)  

        self.mask = torch.nn.Parameter(torch.ones(self.size)).requires_grad_(False)
        self.og_grid = torch.einsum('i,j->ij', torch.ones(self.size), torch.linspace(grid_range[0], grid_range[1], steps=grid_size + 1))
        self.og_grid = torch.nn.Parameter(self.og_grid).requires_grad_(False)
        self.weight_sharing =  torch.arange(self.size)
        self.coef = torch.nn.Parameter(torch.randn(self.size, self.grid_size + self.spline_order))
        noises = (torch.rand(self.size, self.og_grid.shape[1]) - 1 / 2) * scale_noise / grid_size
        
        self.coef = torch.nn.Parameter(self.curve2coef(self.og_grid, noises, self.og_grid, spline_order))
        ##################
        ###            ###
        ##################


    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        # assert x.dim() == 2 and x.size(1) == self.in_features

        # print(f"base_activation: {self.base_activation(x).shape}")
        # print(f"base_weight: {self.base_weight.shape}")
        # print(f"b_splines: {self.b_splines(x).shape}")
        # print(f"scaled_spline_weight: {self.scaled_spline_weight.shape}")

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        # print(f"base_output: {base_output.shape}")
        # print(f"spline_output: {spline_output.shape}")
        # print(f"base_output + spline_output: {(base_output + spline_output).shape}")
        return base_output + spline_output
    
    def get_preacts_postacts(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_features)).reshape(batch, self.size).permute(1, 0)
        preacts = x.permute(1, 0).clone().reshape(batch, self.out_features, self.in_features)
        base = self.base_activation(x).permute(1, 0)
    
        y = self.coef2curve(x_eval=x, grid=self.og_grid[self.weight_sharing], coef=self.coef[self.weight_sharing], k=self.spline_order)  # shape (size, batch)
        y= y.permute(1, 0)

        base_weight = self.base_weight.reshape((1,self.size))
        spline_scaler = self.spline_scaler.reshape((1,self.size))
        # print(base_weight.unsqueeze(dim=0).shape)

        y = base_weight * base + spline_scaler * y
  
        # y = self.scale_bases.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y

        postacts = y.clone().reshape(batch, self.out_features, self.in_features)
        
        return preacts, postacts
    
    def B_batch(self, x, grid, k=0, extend=True, device='cpu'):
        '''
        evaludate x on B-spline bases
        
        Args:
        -----
            x : 2D torch.tensor
                inputs, shape (number of splines, number of samples)
            grid : 2D torch.tensor
                grids, shape (number of splines, number of grid points)
            k : int
                the piecewise polynomial order of splines.
            extend : bool
                If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
            device : str
                devicde
        
        Returns:
        --------
            spline values : 3D torch.tensor
                shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.
        
        Example
        -------
        >>> num_spline = 5
        >>> num_sample = 100
        >>> num_grid_interval = 10
        >>> k = 3
        >>> x = torch.normal(0,1,size=(num_spline, num_sample))
        >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
        >>> B_batch(x, grids, k=k).shape
        torch.Size([5, 13, 100])
        '''

        # x shape: (size, x); grid shape: (size, grid)
        def extend_grid(grid, k_extend=0):
            # pad k to left and right
            # grid shape: (batch, grid)
            h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

            for i in range(k_extend):
                grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
                grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
            grid = grid.to(device)
            return grid

        if extend == True:
            grid = extend_grid(grid, k_extend=k)

        grid = grid.unsqueeze(dim=2).to(device)
        x = x.unsqueeze(dim=1).to(device)

        if k == 0:
            value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
        else:
            B_km1 = self.B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
            value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                        grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
        return value
    
    def coef2curve(self, x_eval, grid, coef, k, device="cpu"):
        '''
        converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
        
        Args:
        -----
            x_eval : 2D torch.tensor)
                shape (number of splines, number of samples)
            grid : 2D torch.tensor)
                shape (number of splines, number of grid points)
            coef : 2D torch.tensor)
                shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
            k : int
                the piecewise polynomial order of splines.
            device : str
                devicde
            
        Returns:
        --------
            y_eval : 2D torch.tensor
                shape (number of splines, number of samples)
            
        Example
        -------
        >>> num_spline = 5
        >>> num_sample = 100
        >>> num_grid_interval = 10
        >>> k = 3
        >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
        >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
        >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
        >>> coef2curve(x_eval, grids, coef, k=k).shape
        torch.Size([5, 100])
        '''
        # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
        # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
        if coef.dtype != x_eval.dtype:
            coef = coef.to(x_eval.dtype)
        y_eval = torch.einsum('ij,ijk->ik', coef, self.B_batch(x_eval, grid, k, device=device))
        return y_eval
    
    def curve2coef(self, x_eval, y_eval, grid, k, device="cpu"):
        '''
        converting B-spline curves to B-spline coefficients using least squares.
        
        Args:
        -----
            x_eval : 2D torch.tensor
                shape (number of splines, number of samples)
            y_eval : 2D torch.tensor
                shape (number of splines, number of samples)
            grid : 2D torch.tensor
                shape (number of splines, number of grid points)
            k : int
                the piecewise polynomial order of splines.
            device : str
                devicde
            
        Example
        -------
        >>> num_spline = 5
        >>> num_sample = 100
        >>> num_grid_interval = 10
        >>> k = 3
        >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
        >>> y_eval = torch.normal(0,1,size=(num_spline, num_sample))
        >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
        torch.Size([5, 13])
        '''
        # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
        mat = self.B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
        # coef = torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
        coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                                driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
        return coef.to(device)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.acts_scale=[]

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        

    def forward(self, x: torch.Tensor, update_grid=False):

        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
  
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    
    def train(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid_=True, grid_update_num=10, loss_fn=None, lr=1., stop_grid_update_step=50, batch=-1,
              small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, sglr_avoid=False, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', device='cpu'):
        '''
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            stop_grid_update_step : int
                no grid updates after this training step
            batch : int
                batch size, if -1 then full.
            small_mag_threshold : float
                threshold to determine large or small numbers (may want to apply larger penalty to smaller numbers)
            small_reg_factor : float
                penalty strength applied to small factors relative to large factos
            device : str
                device   
            save_fig_freq : int
                save figure every (save_fig_freq) step

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization

        Example
        -------
        >>> # for interactive examples, please see demos
        >>> from utils import create_dataset
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
        >>> model.plot()
        '''

        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.layers)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.layers[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.layers[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
            else:
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
            # reg_ = reg(self.acts_scale)
            reg_ = self.regularization_loss(regularize_activation=lamb_l1, regularize_entropy=lamb_entropy)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for _ in pbar:

            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid_:
                # self.update_grid(dataset['train_input'][train_id].to(device))
                self.forward(dataset['train_input'][train_id].to(device), update_grid=True)

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id].to(device))
                if sglr_avoid == True:
                    id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                    train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
                else:
                    train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
                # reg_ = reg(self.acts_scale)
                reg_ = self.regularization_loss( regularize_activation=lamb_l1, regularize_entropy=lamb_entropy)

                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))

            if _ % log == 0:
                pbar.set_description("train loss: %.2e | test loss: %.2e | reg: %.2e " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

        return results


