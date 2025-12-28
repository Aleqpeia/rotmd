import numpy as np
from twodanalysis import MembProp


class OrderParameters:

    @classmethod
    def individual_order_sn1(self, sel, n_chain, atoms_inv=False):
        """

        Code to loop over the number of carbons in the lipid tail and get a list with the carbon and its
        hydrogens for each carbon in the lipid tail: :math:`[C3i, HiX, HiY, ...]`. This list is passed to get_vectors which
        return the averages of each i-th carbon. This code returns an array of dim n_chain with the mean :math:`\braket{cos(\theta_i)^2}`

        Parameters
        ----------
        lipid : str
            Name of the lipid to compute the order parameters
        n_chain : int
            Number of carbons in the lipid tail sn1

        Returns
        -------
        chains : ndarray
            Vector dim n_chains with the mean :math:`\braket{cos(\theta_i)^2}`

        Notes
        -----
        The return is a vector containing the value :math:`\braket{cos^2(\theta_i}`. As follows:

        .. math:: [\braket{cos^2(\theta_2}, \braket{cos^2(\theta_3}, ..., \braket{cos^2(\theta_{n_chain}}]

        The index starts at 2 because that is the carbon the lipid tail starts with.

        """

        # Define list to store the chain cos^2(theta)
        chains = []

        # Loop over carbons
        if not atoms_inv:
            for i in range(n_chain):
                # Define selections for H and C in the chain
                # print(f"Value of the chain {i} sn1")

                selections = [
                    f"name C3{i+2}",
                    f"name H{i+2}X and not name HX",
                    f"name H{i+2}Y and not name HY",
                    f"name H{i+2}Z and not name HZ",
                ]
                # print(selections)

                # Define a list to store atoms
                lista = []

                for selection in selections:
                    atoms = sel.select_atoms(selection)

                    if atoms.n_atoms != 0:
                        lista.append(atoms)
                # Call get_individual that computes the cos^2(theta) for each carbon.
                chains.append(self.get_individual(lista))

                # print(i, self.get_individual(lista).shape, self.get_individual(lista))
            chains = np.array(chains)  # Expect array of dim (n_chain, n_lipids)

        else:

            for atom_list in atoms_inv:
                selections = [f"name {atom_list[i]}" for i in range(len(atom_list))]

                lista = []
                for selection in selections:
                    atoms = sel.select_atoms(selection)
                    if atoms.n_atoms != 0:
                        lista.append(atoms)

                chains.append(self.get_individual(lista))
            chains = np.array(chains)  # Expect array of dim (n_chain, n_lipids)

        return chains

    @staticmethod
    def get_individual(lista):
        r"""This function gets a list with a specific carbon (e.g. C34 or C22)
        and its respective hydrogen (e.g. H4X, H4Y). It computes the vectors
        that connect the carbons and the hydrogens and computes the :math:`cos(\theta)^2`, where :math:`\theta` is
        the angle between each vector and the z-axis. Finally, this function returns a vector with the individual (per lipid)
        :math:`\braket{cos(\theta)^2}`, where the mean is computed over the hydrogens of each carbon.

        Parameters
        ----------
        lista : list
            Vector of the shape :math:`[C*i, HiX,HiY, HiZ]`, the minimum len is 2 (when the carbon
            only have one hydrogen) and the maximum is 4 (when there is three hydrogens)
            Note: If there is N lipids, there will be N carbons :math:`C*i`, and the i represents
            the position of the carbon in the lipid tail.

        Returns
        -------
        order : array(n_lipids)
            Float with the mean of :math:`\braket{cos(\theta)^2}`

        Notes
        -----
        The average of the angle of the i-th carbon for all the lipids in the selection is computed
        as follows:

        .. math:: \braket{cos(\theta_i)^2}

        where :math:`\theta_i` is the angle between the z- axis and the vector that connects the i-th carbon and the hydrogen.



        """

        angles = []  # Store the angles for the working carbon
        for i in range(
            len(lista) - 1
        ):  # Accounts for variable number of list (Change if the carbon has or not double bonds)
            vectores = (
                lista[i + 1].positions - lista[0].positions
            )  # Hidrogen - Carbons; output of shape (n_lipids, 3)
            costheta = (
                vectores[:, 2] ** 2 / np.linalg.norm(vectores, axis=1) ** 2
            )  # Compute the costheta^2
            angles.append(costheta)  # dim (n_lipids,)
        angles = np.array(angles)  # dim ((1, 2 or 3),n_lipids)
        # print("angles", angles.shape)
        angles = np.mean(
            angles, axis=0
        )  # output is dim n_lipids, it means the cos^2(theta) or the Carbon passed for each lipid
        return angles

    # Get the cos^2(theta) for each carbon in the selection, for sn2
    @classmethod
    def individual_order_sn2(self, sel, n_chain, atoms_inv=False):
        r"""

        Code to loop over the number of carbons in the lipid tail and get a list with the carbon and its
        hydrogens for each carbon in the lipid tail: :math:`[C2i, HiX, HiY, ...]`. This list is passed to get_vectors which
        return the averages of each i-th carbon. This code returns an array of dim n_chain with the mean :math:`\braket{cos(\theta_i)^2}`

        Parameters
        ----------
        lipid : str
            Name of the lipid to compute the order parameters
        n_chain : int
            Number of carbons in the lipid tail sn1

        Returns
        -------
        chains : ndarray
            Vector dim n_chains with the mean :math:`\braket{cos(\theta_i)^2}`

        Notes
        -----
        The return is a vector containing the value :math:`\braket{cos^2(\theta_i}`. As follows:


        .. math:: [\braket{cos^2(\theta_2}, \braket{cos^2(\theta_3}, ..., \braket{cos^2(\theta_{n_chain}}]

        The index starts at 2 because that is the carbon the lipid tail starts with.

        """
        # Define list to store the chain cos^2(theta)
        chains = []
        # Loop over carbons
        if not atoms_inv:
            for i in range(n_chain):
                # Define selections for H and C in the chain
                # print(f"Value of the chain {i} sn2")
                selections = [
                    f"name C2{i+2}",
                    f"name H{i+2}R and not name HR",
                    f"name H{i+2}S and not name HS",
                    f"name H{i+2}T and not name HT",
                ]
                # if lipid == "POPE" or lipid == "POPS" or lipid == "POPI15" or lipid == "POPI24":
                #    if selections[0] == "name C29":
                #        selections[1] = "name H91"
                #    if selections[0] == "name C210":
                #        selections[1] = "name H101"
                # Define a list to store atoms
                lista = []

                for selection in selections:
                    atoms = sel.select_atoms(selection)
                    if atoms.n_atoms != 0:
                        lista.append(atoms)
                if len(lista) == 1:

                    one_atom = f"name H{i+2}1"

                    atoms = sel.select_atoms(one_atom)
                    if atoms.n_atoms != 0:
                        lista.append(atoms)
                if len(lista) == 1:
                    raise "Something went wrong guessing the atoms in lipid tail"

                angles = self.get_individual(lista)
                chains.append(angles)
            chains = np.array(chains)  # Expect array of dim (n_chain, n_lipids)
        else:

            for atom_list in atoms_inv:
                selections = [f"name {atom_list[i]}" for i in range(len(atom_list))]

                lista = []
                for selection in selections:
                    atoms = sel.select_atoms(selection)
                    if atoms.n_atoms != 0:
                        lista.append(atoms)
                chains.append(self.get_individual(lista))
            chains = np.array(chains)  # Expect array of dim (n_chain, n_lipids)
        return chains

    def _check_one_lipid(atoms):
        resnames = list(set(atoms.residues.resnames))
        if len(resnames) != 1:
            raise "This function only works for one lipid at a time. Please make sure you use only one lipid"
        return resnames[0]

    @classmethod
    def scd(
        self,
        universe,
        selection,
        chain="sn1",
        start=0,
        final=-1,
        step=1,
        n_chain=None,
        connection_chains=None,
    ):

        lipids = universe.select_atoms(selection)
        lipid_name = self._check_one_lipid(lipids)

        membrane = MembProp(lipids, connection_chains=connection_chains)

        if n_chain is None:
            membrane.guess_chain_lenght()
            chain_info = membrane.chain_info
            if chain == "sn1":
                n_chain = chain_info[lipid_name][0]
            elif chain == "sn2":
                n_chain = chain_info[lipid_name][1]
            else:
                raise "This code only support sn1 and sns as chains"
        chain_structure = membrane.extract_chain_info(lipid_name)
        carbons = {
            "sn1": [
                chain_structure[1][i][k]
                for i in range(len(chain_structure[1]))
                for k in range(len(chain_structure[1][i]))
                if "C" in chain_structure[1][i][k]
            ],
            "sn2": [
                chain_structure[0][i][k]
                for i in range(len(chain_structure[0]))
                for k in range(len(chain_structure[0][i]))
                if "C" in chain_structure[0][i][k]
            ],
        }

        orders = []
        for _ in universe.trajectory[start:final:step]:

            if chain == "sn1":
                order_data = self.individual_order_sn1(
                    lipids, n_chain, atoms_inv=chain_structure[1]
                )
            elif chain == "sn2":
                order_data = self.individual_order_sn2(
                    lipids, n_chain, atoms_inv=chain_structure[0]
                )

            orders.append(np.mean(order_data, axis=1))
        orders = np.array(orders)
        final = np.abs(1.5 * np.mean(orders, axis=0) - 0.5)

        return [carbons[chain], final]

    @classmethod
    def window_scd(
        self,
        universe,
        selection,
        start,
        final,
        window,
        chain,
        step=1,
        connection_chains=None,
    ):

        length = len(universe.trajectory[start:final:step])
        n_windows = length // window
        values = []

        for i in range(n_windows):
            start_local = start + i * window
            final_local = start + (i + 1) * window
            carbons, scd1 = self.scd(
                universe,
                selection,
                chain,
                start=start_local,
                final=final_local,
                step=1,
                connection_chains=connection_chains,
            )

            values.append(scd1)

        values = np.array(values)
        error = np.std(values, axis=0)
        values = np.mean(values, axis=0)

        return [carbons, values, error]


# class SplayAngle:
