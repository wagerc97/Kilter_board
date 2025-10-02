
from collections import Counter
from typing import Dict, Tuple
import sqlite3
import pandas as pd
from pathlib import Path
import numpy as np

from torch.utils.data import  DataLoader, Dataset, random_split
import torch
from pympler import asizeof
import torch.nn.functional as F
import sys
import kornia.filters as KF
import os


class ClimbingDataset(Dataset):
    def __init__(self,
                board_names: list[str] = ["12 x 12 with kickboard Square"],
                map: bool = False,
                transform=None,
                max_samples: int | None = None,
                label_filter: list[int] = None) -> None:
        """
        Custom PyTorch Dataset for climbing board problems.

        Loads route data from one or more boards into a unified dataset, 
        supports filtering by difficulty, optional coordinate remapping,
        and applying transforms (e.g. Kornia augmentations).

        Args:
            board_names (list[str], optional): List of board names to load 
                (must exist in the database). Defaults to ["12 x 12 with kickboard Square"].
            map (bool, optional): If True, remap x/y coordinates into compact
                integer indices (no gaps). Defaults to False.
            transform (callable, optional): Transform function applied to each 
                sample (e.g. Kornia.GaussianBlur2d). Defaults to None.
            label_filter (list[int], optional): If provided, keep only routes
                whose difficulty is in this list. Defaults to None.

        Attributes:
            data (pd.DataFrame): Raw data loaded and preprocessed from boards.
            max_x_value (int): Maximum x-coordinate (after optional remap).
            max_y_value (int): Maximum y-coordinate (after optional remap).
            num_classes (int): Number of unique difficulty levels.
            unique_routs (np.ndarray): Sorted array of unique route IDs.
            num_routs (int): Total number of routes.
            route_groups (dict[int, pd.DataFrame]): Dictionary mapping route_id 
                to its corresponding DataFrame slice for fast lookup.
        """

        # Initialize empty DataFrame and save config
         
        self.board_names = board_names
        self.map = map
        self.transform = transform
        self.max_samples = max_samples

        # Load data for each board and concatenate
        self.data_df = ClimbingDataset.load_df(board_names=board_names)

        
        # Optionally remap x and y values into a dense integer grid
        if map:
            self.data_df = self.map_xy_values()

        # Save maximum board dimensions (after mapping, if applied)
        self.max_x_value = self.data_df["x"].values.max()
        self.max_y_value = self.data_df["y"].values.max()

        # Normalize difficulty labels (subtract min to start from 0)
        self.data_df["difficulty"] = self.map_labels()
        self.num_classes = len(self.data_df["difficulty"].unique()) 

        # Optionally filter dataset by difficulty levels
        if label_filter:
            self.data_df = self.data_df[self.data_df["difficulty"].isin(label_filter)].reset_index(drop=True)

        # Extract unique route IDs and count routes
        self.unique_routs = np.sort(self.data_df["route_id"].unique())
        self.num_routs = len(self.unique_routs)

        # Precompute route_id → DataFrame mapping for fast access in __getitem__
        self.route_groups = {rid: df for rid, df in self.data_df.groupby("route_id")}
        
        if self.max_samples is not None:
            self.reduce_dataset(max_samples= self.max_samples)
                
        
        # precompute the spae mtx not higth storage demand fast acces entier datadt shoult that 5.5 GB RAM :/
        self.routs_mtx = []
        for route_id, route_df in self.route_groups.items():
            X, y = ClimbingDataset.build_route_matrix(
                route_df, 
                mtx_encoding=self.mtx_encoding, 
                num_classes=self.num_classes, 
                transform=self.transform
            )
            self.routs_mtx.append((X, y))
            
        self.num_features = X.shape[0]
        # Delet overhead 
        del self.data_df
        #del self.route_groups
        
        
        # Warn if memory usage is very high (>4 GB)
        #storage = sys.getsizeof(self.routs_mtx) / 1e6

        #print(f"| USED STORAGE: {storage:.3f} MB")
            
            
    
    def reduce_dataset(self, max_samples: int):
        """
        Reduce the dataset to contain at most `max_samples` routes.

        This method modifies the object in-place:
        - Truncates `self.unique_routs` to the first `max_samples`.
        - Reduces `self.route_groups` to only include the selected routes.
        - Updates `self.num_routs`.

        Parameters
        ----------
        max_samples : int
            Maximum number of routes to keep. Must be less than the current
            number of routes.
        """
        if max_samples < self.num_routs:
            limited_ids = self.unique_routs[:max_samples]
            self.route_groups = {rid: self.route_groups[rid] for rid in limited_ids}
            self.unique_routs = limited_ids
            self.num_routs = max_samples
    

    def mtx_encoding(self, df):
        # Convert numpy arrays -> torch without extra copy
        x = torch.from_numpy(df["x"].to_numpy())
        y = torch.from_numpy(df["y"].to_numpy())
        # Build matrix on GPU directly
        mtx = torch.zeros(
            int(self.max_y_value) + 1,
            int(self.max_x_value) + 1,
            dtype=torch.float32
        )
        # Scatter ones
        mtx[y, x] = 1.0
        return mtx
            

    def __getitem__(self, index: int):
        """
        Retrieve a single climbing route sample.
        Returns (X, y) = (4-channel hold matrix, one-hot label).
        """
        return self.routs_mtx[index]

    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.num_routs



    def map_labels(self):
        """Normalize difficulty labels by subtracting the minimum value."""
        min_diff = self.data_df["difficulty"].min() 
        return self.data_df["difficulty"] - min_diff
        
            
    def map_xy_values(self):
        """
        Map the original x and y coordinate values in `self.data_df` to
        sequential integer indices based on their sorted unique values.

        This is useful when the raw coordinates are arbitrary or sparse,
        but you want to normalize them into a compact grid representation.

        Returns:
            pandas.DataFrame:
                A copy of `self.data_df` where the 'x' and 'y' columns are replaced
                by integer indices (0-based). All other columns are unchanged.
        """
        unique_x = sorted(self.data_df["x"].unique())
        unique_y = sorted(self.data_df["y"].unique())
        
        x_map = {val: idx for idx, val in enumerate(unique_x)}
        y_map = {val: idx for idx, val in enumerate(unique_y)}

        data = self.data_df.copy()
        data["x"] = self.data_df["x"].map(x_map)
        data["y"] = self.data_df["y"].map(y_map)
        
        return data
        
        
                
    @staticmethod
    def build_route_matrix(
                    route_df, 
                    mtx_encoding, 
                    num_classes: int, 
                    transform=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the feature matrix (X) and label (y) for a single climbing route.
        Static to enable paralelisation

        Parameters
        ----------
        route_df : pandas.DataFrame
            DataFrame containing holds and metadata for a single route.
        mtx_encoding : callable
            Function to encode a subset of the DataFrame (e.g. holds with a given role).
        num_classes : int
            Total number of difficulty classes (for one-hot encoding).
        transform : callable, optional
            Transformation applied to the feature matrix X.

        Returns
        -------
        X : torch.Tensor
            Encoded feature tensor of shape (1, 4, H, W) (example shape).
        y : torch.Tensor
            One-hot encoded difficulty label tensor of shape (num_classes,).
        """
        # label
        difficulty = route_df["difficulty"].unique()[0]
        y = F.one_hot(torch.tensor(difficulty), num_classes=num_classes).float()

        # inputs
        start  = mtx_encoding(route_df[route_df["role_name"] == "start"])
        middle = mtx_encoding(route_df[route_df["role_name"] == "middle"])
        foot   = mtx_encoding(route_df[route_df["role_name"] == "foot"])
        finish = mtx_encoding(route_df[route_df["role_name"] == "finish"])

        
        X = torch.stack([start, middle, foot, finish], dim=0).unsqueeze(0)
        

        if transform:
            X = transform(X).squeeze() 

        return X, y
    
    
    
    @staticmethod
    def load_df(board_names: list[str]) -> pd.DataFrame:
        """
        Load and concatenate climbing datasets for the given boards.

        Parameters
        ----------
        board_names : list of str
            A list of board names for which data should be loaded.

        Returns
        -------
        pd.DataFrame
            A single DataFrame containing concatenated data from all specified boards.
        """
        df = pd.DataFrame()
        for each_board in board_names:
            data_board = ClimbingDataset.get_data_custom(board_name=each_board)
            df = pd.concat([df, data_board], ignore_index=True)
        return df
        

    @staticmethod
    def get_all_routes_with_tokens(board_id:int, conn=None, limit: int | None = None):
        base_sql = """
            SELECT
            b.id    AS route_id,
            b.difficulty,
            CAST(SUBSTR(t.token, 2) AS INTEGER) AS token_num,

            -- add token type directly in SQL
            CASE
                WHEN t.token LIKE 'p%' THEN 'placement'
                WHEN t.token LIKE 'r%' THEN 'role'
                ELSE 'unknown'
            END AS token_type,

            -- placement info (only if token starts with 'p')
            p.x, p.y, p.set_id,

            -- role info (only if token starts with 'r')
            r.name  AS role_name,
            r.color AS role_color

            FROM board_samples b
            JOIN json_each(b.holds) h
            LEFT JOIN static.id_to_token t
            ON t.id = h.value
            LEFT JOIN static.placements p
            ON p.token_id = CAST(SUBSTR(t.token, 2) AS INTEGER)
            AND t.token LIKE 'p%'
            LEFT JOIN static.roles r
            ON r.role_id = CAST(SUBSTR(t.token, 2) AS INTEGER)
            AND t.token LIKE 'r%'
            WHERE b.board_id = ?
        """
        if limit is None:
            sql = base_sql  # no LIMIT
            params = [board_id]
        else:
            sql = base_sql + " LIMIT ?"
            params = [board_id, limit]
        return pd.read_sql(sql, conn, params=params)
    
    @staticmethod
    def get_data_custom(board_name: str, difficulty: int | None = None, limit: int | None = None) -> pd.DataFrame:
        """
        Load placements and roles for a given board.

        Args:
            board_name: The name of the board (as stored in static.token_to_id).
            difficulty: Optional filter for route difficulty.
            limit: Optional row limit (None = no limit).

        Returns:
            A dict with:
            - 'placements': DataFrame of placement tokens with coordinates
            - 'roles': DataFrame of role tokens with role info
        """
        
        base_dir = Path(__file__).resolve().parent.parent   # project_root/
        data_dir = base_dir / "data"

        boards_db = data_dir / "boards.db"
        static_db = data_dir / "static.db"
        

        with sqlite3.connect(boards_db) as conn:
            # attach static db
            conn.execute(f"ATTACH DATABASE '{static_db}' AS static;")

            # get board_id from name
            board_id_df = pd.read_sql(
                "SELECT id FROM static.token_to_id WHERE token = ?",
                conn,
                params=[board_name]
            )
            if board_id_df.empty:
                raise ValueError(f"Board '{board_name}' not found in token_to_id")
            board_id = int(board_id_df.iloc[0]["id"])

            # query routes with placements + roles
            df = ClimbingDataset.get_all_routes_with_tokens(board_id=board_id, conn=conn, limit=limit)

            if difficulty is not None:
                df = df[df.difficulty == difficulty]
                
            

            placements = df[df.token_type == "placement"].reset_index(drop=True)
            roles      = df[df.token_type == "role"].reset_index(drop=True)
            
            roles = roles.drop(columns=["route_id", "difficulty", "token_num", "token_type", "x", "y", "set_id"])
            placements = placements.drop(columns=["token_num", "token_type", "role_name", "role_color"])
            data = roles.join(placements, how="inner", lsuffix="_left", rsuffix="_right")

        return data


def CNN_dataloaders(
                board_names: list[str] = ["12 x 12 with kickboard Square"],
                map: bool = True,
                max_samples: int | None = None,
                label_filter: list[int] = [5, 14],      # 6a+ & 7c
                blur_kernel_size:int = 3,
                blur_sigma: float = 1.0,
                batch_size:int = 32,
                num_workers:int = 0,
                train_test_split:float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test DataLoaders for climbing board data.

    This function builds a dataset using `ClimbingDataset`, applies Gaussian blur
    as a preprocessing transform, splits the dataset into train/test subsets, 
    and returns corresponding PyTorch DataLoaders.

    Parameters
    ----------
    board_names : list of str, default=["12 x 12 with kickboard Square"]
        Names of climbing boards to include in the dataset.
    map : bool, default=True
        Whether to return the climbing problems as maps.
    max_samples : int or None, default=None
        Maximum number of samples to include. If None, use the full dataset.
    label_filter : list of int, default=[5, 14]
        List of labels (difficulty levels) to keep in the dataset.
    blur_kernel_size : int, default=3
        Size of the Gaussian blur kernel.
    blur_sigma : float, default=1.0
        Standard deviation for the Gaussian blur.
    batch_size : int, default=32
        Number of samples per batch in the DataLoader.
    num_workers : int, default=0
        Number of subprocesses used for data loading. 
        On Windows, often set to 0 for stability.
    train_test_split : float, default=0.8
        Fraction of data to use for training (rest is used for testing).
        Must be between 0.0 and 1.0. Invalid values are reset to 0.8.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set.

    Notes
    -----
    - The train/test split is deterministic with a fixed random seed (42).
    - Gaussian blur is applied as a preprocessing step to all samples.
    """
    
    transform = KF.GaussianBlur2d(kernel_size=(blur_kernel_size, blur_kernel_size), sigma=(blur_sigma, blur_sigma), border_type="constant")
    
    dataset = ClimbingDataset(board_names=board_names,
                                map=map, 
                                transform=transform,
                                label_filter=label_filter,
                                max_samples=max_samples
                                
    )
    
    if not (0.0 <= train_test_split <= 1.0):
        print(f"Invalid split {train_test_split}, reset to 0.8")
        train_test_split = 0.8
    
    train_size = int(train_test_split * len(dataset))  # 80% train
    test_size = len(dataset) - train_size

    # Deterministic split (set generator seed for reproducibility)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader
    








# Flat table
def get_all_routes_with_tokens(board_id:int, conn=None, limit: int | None = None):
    base_sql = """
    SELECT
      b.id    AS route_id,
      b.difficulty,
      h.value AS hold_id,
      t.token AS hold_token,
      CAST(SUBSTR(t.token, 2) AS INTEGER) AS token_num,

      -- add token type directly in SQL
      CASE
        WHEN t.token LIKE 'p%' THEN 'placement'
        WHEN t.token LIKE 'r%' THEN 'role'
        ELSE 'unknown'
      END AS token_type,

      -- placement info (only if token starts with 'p')
      p.x, p.y, p.set_id,

      -- role info (only if token starts with 'r')
      r.name  AS role_name,
      r.color AS role_color

    FROM board_samples b
    JOIN json_each(b.holds) h
    LEFT JOIN static.id_to_token t
      ON t.id = h.value
    LEFT JOIN static.placements p
      ON p.token_id = CAST(SUBSTR(t.token, 2) AS INTEGER)
     AND t.token LIKE 'p%'
    LEFT JOIN static.roles r
      ON r.role_id = CAST(SUBSTR(t.token, 2) AS INTEGER)
     AND t.token LIKE 'r%'
    WHERE b.board_id = ?
    """
    if limit is None:
        sql = base_sql  # no LIMIT
        params = [board_id]
    else:
        sql = base_sql + " LIMIT ?"
        params = [board_id, limit]
    return pd.read_sql(sql, conn, params=params)



def get_data(board_name: str, difficulty: int | None = None, limit: int | None = None) -> Dict[str, pd.DataFrame]:
    """
    Load placements and roles for a given board.

    Args:
        board_name: The name of the board (as stored in static.token_to_id).
        difficulty: Optional filter for route difficulty.
        limit: Optional row limit (None = no limit).

    Returns:
        A dict with:
          - 'placements': DataFrame of placement tokens with coordinates
          - 'roles': DataFrame of role tokens with role info
    """
    # Get the path to the current file (e.g. kilter/)
    base_dir = Path(__file__).resolve().parent.parent

    # Data folder inside kilter/
    data_dir = base_dir / "data"

    boards_db = data_dir / "boards.db"
    static_db = data_dir / "static.db"
    
    #print(f"Loading data for board '{board_name}' from {boards_db} and {static_db}")

    with sqlite3.connect(boards_db) as conn:
        # attach static db
        conn.execute(f"ATTACH DATABASE '{static_db}' AS static;")

        # get board_id from name
        board_id_df = pd.read_sql(
            "SELECT id FROM static.token_to_id WHERE token = ?",
            conn,
            params=[board_name]
        )
        if board_id_df.empty:
            raise ValueError(f"Board '{board_name}' not found in token_to_id")
        board_id = int(board_id_df.iloc[0]["id"])

        # query routes with placements + roles
        df = get_all_routes_with_tokens(board_id=board_id, conn=conn, limit=limit)

        if difficulty is not None:
            df = df[df.difficulty == difficulty]
            
        

        placements = df[df.token_type == "placement"].reset_index(drop=True)
        roles      = df[df.token_type == "role"].reset_index(drop=True)

    return {"placements": placements, "roles": roles}













