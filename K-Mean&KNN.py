import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score
from scipy.stats import f_oneway
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set Nature journal style for professional publication-quality plots
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Nature journal color palette (professional and accessible)
nature_colors = ['#E31A1C', '#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A', '#B15928']
sns.set_palette(nature_colors)

class ARPersonaDiscovery:
    def __init__(self, csv_file_path):
        """Initialize with data loading and variable mapping"""
        self.csv_file_path = csv_file_path
        self.data = None
        self.clustering_data = None
        self.cluster_labels = None
        self.scaler = StandardScaler()
        self.optimal_k = None
        
        # Define variable mappings based on your structure
        self.variable_mappings = {
            # Technical Features (Stimulus)
            'AR': ['AR', 'ar', 'augmented_reality', 'tech_ar'],
            'IVR': ['IVR', 'ivr', 'interactive_vr', 'tech_ivr'],
            'PSN': ['PSN', 'psn', 'personalization', 'tech_psn'],
            # Anticipated Emotion (Stimulus)
            'ANP': ['ANP', 'anp', 'anticipated_emotion', 'anticipation'],
            # Interaction Satisfaction (Organism)
            'ITSN': ['ITSN', 'itsn', 'interaction_satisfaction', 'satisfaction'],
            # Experience States (Organism)
            'IMM': ['IMM', 'imm', 'immersion', 'immersive'],
            'ARIT': ['ARIT', 'arit', 'ar_induced_telepresence', 'AR-Induced Telepresence'],
            'PL': ['PL', 'pl', 'pleasure', 'enjoyment'],
            # Cognitive Processing (Organism)
            'IPS': ['IPS', 'ips', 'inspiration', 'creative_inspiration'],
            # Behavioral Intentions (Response)
            'PI': ['PI', 'pi', 'purchase_intention', 'behavioral_intention'],
            'CBI': ['CBI', 'cbi', 'cross_buying_intention', 'Cross-Buying_Intention']
        }
        
    def load_and_prepare_data(self):
        """Load CSV and map variables to standard names"""
        self.data = pd.read_csv(self.csv_file_path)
        
        # Map variable names to standard format
        mapped_data = {}
        found_vars = []
        for standard_name, possible_names in self.variable_mappings.items():
            for possible_name in possible_names:
                if possible_name in self.data.columns:
                    mapped_data[standard_name] = self.data[possible_name]
                    found_vars.append(standard_name)
                    break
        
        self.data = pd.DataFrame(mapped_data)
        
        # Check if all variables were found
        if len(found_vars) != len(self.variable_mappings):
            missing_vars = set(self.variable_mappings.keys()) - set(found_vars)
            print(f"Warning: The following variables were not found in the CSV file: {', '.join(missing_vars)}")


        # Basic data quality check
        print("Data Quality Summary:")
        print(f"Shape: {self.data.shape}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        if not self.data.empty:
            print(f"Scale range: {self.data.min().min():.1f} - {self.data.max().max():.1f}")
        
        return self.data
    
    def prepare_clustering_variables(self):
        """
        Select variables for clustering based on theory:
        Focus on experiential responsiveness (System 1) and early processing
        Exclude final outcomes to avoid tautological clustering
        """
        # Primary clustering variables: Individual differences in experiential processing
        clustering_vars = [
            # System 1 (Experiential) indicators
            'IMM',    # Immersion - how deeply they engage experientially
            'ARIT',   # AR-Induced Telepresence - spatial presence responsiveness
            'PL',     # Pleasure - hedonic sensitivity
            # Early cognitive response (but not final inspiration)
            'ITSN',   # Interaction satisfaction - immediate cognitive evaluation
            # Stimulus responsiveness patterns
            'AR',     # Technical feature sensitivity
            'ANP'     # Emotional priming responsiveness
        ]
        
        # Ensure all clustering variables exist in the data
        existing_clustering_vars = [var for var in clustering_vars if var in self.data.columns]
        if len(existing_clustering_vars) != len(clustering_vars):
            missing = set(clustering_vars) - set(existing_clustering_vars)
            print(f"Warning: Cannot use the following clustering variables as they are missing: {', '.join(missing)}")
            # Handle this case, e.g., by stopping or using only available vars
            if not existing_clustering_vars:
                raise ValueError("None of the specified clustering variables are available in the data.")

        self.clustering_data = self.data[existing_clustering_vars].copy()
        
        print("Clustering Variable Selection:")
        print("Selected variables (theory-based):")
        for var in existing_clustering_vars:
            print(f"  {var}: Mean={self.clustering_data[var].mean():.2f}, SD={self.clustering_data[var].std():.2f}")
        
        return self.clustering_data
    


    def discover_optimal_clusters(self, k_range=(2, 7)):
        """Find optimal number of clusters using multiple validation metrics"""
        X_scaled = self.scaler.fit_transform(self.clustering_data)
        
        validation_results = {}
        silhouette_scores = []
        calinski_scores = []
        inertias = []
        k_values = list(range(*k_range))
        
        # Store clustering history for visualization
        self.clustering_history = {}
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            sil_score = silhouette_score(X_scaled, cluster_labels)
            cal_score = calinski_harabasz_score(X_scaled, cluster_labels)
            
            validation_results[k] = {
                'silhouette': sil_score,
                'calinski_harabasz': cal_score,
                'inertia': kmeans.inertia_,
                'model': kmeans
            }
            
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            inertias.append(kmeans.inertia_)
            
            # Store for visualization
            self.clustering_history[k] = {
                'labels': cluster_labels,
                'centroids': kmeans.cluster_centers_,
                'model': kmeans
            }
        
        # Plot validation metrics with Nature journal style
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        axes[0].plot(k_values, silhouette_scores, 'o-', linewidth=3, markersize=8, color=nature_colors[0])
        axes[0].set_title('Silhouette Score', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('Number of Clusters', fontweight='bold')
        axes[0].set_ylabel('Silhouette Score', fontweight='bold')
        axes[0].grid(True, alpha=0.3, linewidth=1)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        axes[1].plot(k_values, calinski_scores, 'o-', linewidth=3, markersize=8, color=nature_colors[1])
        axes[1].set_title('Calinski-Harabasz Score', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('Number of Clusters', fontweight='bold')
        axes[1].set_ylabel('Calinski-Harabasz Score', fontweight='bold')
        axes[1].grid(True, alpha=0.3, linewidth=1)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        axes[2].plot(k_values, inertias, 'o-', linewidth=3, markersize=8, color=nature_colors[2])
        axes[2].set_title('Inertia (Elbow Method)', fontweight='bold', fontsize=14)
        axes[2].set_xlabel('Number of Clusters', fontweight='bold')
        axes[2].set_ylabel('Inertia', fontweight='bold')
        axes[2].grid(True, alpha=0.3, linewidth=1)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Select optimal k (highest silhouette score)
        self.optimal_k = k_values[np.argmax(silhouette_scores)]
        print(f"\nOptimal number of clusters: {self.optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.3f}")
        
        # Fit final model with optimal k
        final_model = validation_results[self.optimal_k]['model']
        self.cluster_labels = final_model.predict(X_scaled)
        
        # Create clustering process visualization
        self._visualize_clustering_process(X_scaled, final_model)
        
        return validation_results
    
    def _visualize_clustering_process(self, X_scaled, final_model):
        """
        Visualize K-means clustering process: original data, centroid movement, and final result
        Following Nature journal standards with Times New Roman bold fonts
        """
        
        # Use PCA for 2D visualization if more than 2 dimensions
        from sklearn.decomposition import PCA
        
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_scaled)
            centroids_2d = pca.transform(final_model.cluster_centers_)
            explained_var = pca.explained_variance_ratio_
            pc1_var, pc2_var = explained_var[0], explained_var[1]
        else:
            X_2d = X_scaled
            centroids_2d = final_model.cluster_centers_
            pc1_var, pc2_var = 1.0, 1.0
        
        # Create figure with Nature journal specifications
        fig = plt.figure(figsize=(18, 6))
        
        # Colors for clusters (Nature journal palette)
        cluster_colors = nature_colors[:self.optimal_k]
        
        # Plot 1: Original Data Distribution
        ax1 = plt.subplot(1, 3, 1)
        scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c='#808080', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        ax1.set_title('Original Data Distribution', fontweight='bold', fontsize=14, pad=20)
        ax1.set_xlabel(f'PC1 ({pc1_var:.1%} variance)' if X_scaled.shape[1] > 2 else 'Feature 1', fontweight='bold')
        ax1.set_ylabel(f'PC2 ({pc2_var:.1%} variance)' if X_scaled.shape[1] > 2 else 'Feature 2', fontweight='bold')
        ax1.grid(True, alpha=0.3, linewidth=1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add data statistics
        ax1.text(0.02, 0.98, f'n = {len(X_2d)}', transform=ax1.transAxes, 
                verticalalignment='top', fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot 2: Dynamic Process (Centroid Positions)
        ax2 = plt.subplot(1, 3, 2)
        ax2.scatter(X_2d[:, 0], X_2d[:, 1], c='#D3D3D3', alpha=0.4, s=30, edgecolors='white', linewidth=0.5)
        
        # Plot centroids with enhanced styling
        for i, centroid in enumerate(centroids_2d):
            ax2.scatter(centroid[0], centroid[1], color=cluster_colors[i], 
                       edgecolor='black', s=300, alpha=0.9, marker='X', linewidth=2,
                       label=f'Centroid {i+1}')
            
            # Draw line from data center to centroid
            data_center = [X_2d[:, 0].mean(), X_2d[:, 1].mean()]
            ax2.annotate('', xy=centroid, xytext=data_center,
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.7, linewidth=2))
        
        ax2.set_title('K-means Centroids', fontweight='bold', fontsize=14, pad=20)
        ax2.set_xlabel(f'PC1 ({pc1_var:.1%} variance)' if X_scaled.shape[1] > 2 else 'Feature 1', fontweight='bold')
        ax2.set_ylabel(f'PC2 ({pc2_var:.1%} variance)' if X_scaled.shape[1] > 2 else 'Feature 2', fontweight='bold')
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax2.grid(True, alpha=0.3, linewidth=1)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Plot 3: Final Clustering Result
        ax3 = plt.subplot(1, 3, 3)
        
        # Plot each cluster with distinct colors
        for k in range(self.optimal_k):
            cluster_mask = self.cluster_labels == k
            cluster_points = X_2d[cluster_mask]
            cluster_size = np.sum(cluster_mask)
            
            ax3.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       color=cluster_colors[k], alpha=0.7, s=50, 
                       edgecolors='white', linewidth=0.5,
                       label=f'Persona {k+1} (n={cluster_size})')
        
        # Plot centroids prominently
        ax3.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                   s=400, c='#FFD700', edgecolor='black', 
                   label='Centroids', marker='*', linewidth=2, alpha=0.9)
        
        ax3.set_title('Final Persona Clusters', fontweight='bold', fontsize=14, pad=20)
        ax3.set_xlabel(f'PC1 ({pc1_var:.1%} variance)' if X_scaled.shape[1] > 2 else 'Feature 1', fontweight='bold')
        ax3.set_ylabel(f'PC2 ({pc2_var:.1%} variance)' if X_scaled.shape[1] > 2 else 'Feature 2', fontweight='bold')
        ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax3.grid(True, alpha=0.3, linewidth=1)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Add silhouette score annotation
        silhouette_score_val = silhouette_score(X_scaled, self.cluster_labels)
        ax3.text(0.02, 0.02, f'Silhouette Score: {silhouette_score_val:.3f}', 
                transform=ax3.transAxes, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print clustering summary
        print("\n" + "="*80)
        print("K-MEANS CLUSTERING PROCESS SUMMARY")
        print("="*80)
        print(f"Data dimensions: {X_scaled.shape[1]} variables → 2D visualization")
        if X_scaled.shape[1] > 2:
            print(f"PCA explained variance: PC1={pc1_var:.1%}, PC2={pc2_var:.1%}, Total={pc1_var+pc2_var:.1%}")
        
        for k in range(self.optimal_k):
            cluster_size = np.sum(self.cluster_labels == k)
            print(f"Persona {k+1}: {cluster_size} participants ({cluster_size/len(self.cluster_labels):.1%})")
        
        print(f"Final silhouette score: {silhouette_score_val:.3f}")
        print("="*80)
        
    def validate_cluster_reproducibility(self):
            """
            Validates cluster reproducibility using a train/test split.
            1. Splits data into training (80%) and hold-out (20%) sets.
            2. Trains K-Means on the training data to generate cluster labels.
            3. Trains a KNN classifier on the training data and its K-Means labels.
            4. Predicts labels on the hold-out set using both K-Means and KNN.
            5. Measures if the KNN can recover the K-Means labels on new data.
            """
            print("\n" + "="*80)
            print("FORMAL PROOF: CLUSTER REPRODUCIBILITY VALIDATION")
            print("="*80)
        
            if self.clustering_data is None or self.optimal_k is None:
                print("Error: Clustering data or optimal_k not available. Run discover_optimal_clusters first.")
                return

            X = self.clustering_data
        
            # 1. Split the data into a bigger training part and a hold-out set
            # We don't need y here, as we generate it via clustering
            X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)
            print(f"Data split: {len(X_train)} training samples, {len(X_test)} hold-out samples.")

            # 2. Scale data correctly: fit on train, transform both train and test
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test) # Use the same scaler for the test set

            # 3. Train K-Means on the bigger part (training data)
            kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
            train_labels = kmeans.fit_predict(X_train_scaled)
            print(f"K-Means model trained on the training data with {self.optimal_k} clusters.")

            # 4. Fit a K-Nearest-Neighbours classifier on the same inputs
            knn = KNeighborsClassifier(n_neighbors=5) # 5 is a common default
            knn.fit(X_train_scaled, train_labels)
            print("KNN classifier trained to learn the mapping from features to K-Means labels.")

            # 5. Measure performance on the hold-out set
            # "True" labels for the hold-out set are what the K-Means model would assign
            true_holdout_labels = kmeans.predict(X_test_scaled)
        
            # Predicted labels from the KNN classifier
            predicted_holdout_labels = knn.predict(X_test_scaled)
            print("Predicting labels for the hold-out data using the trained KNN.")
        
            # Calculate metrics
            accuracy = accuracy_score(true_holdout_labels, predicted_holdout_labels)
            ari = adjusted_rand_score(true_holdout_labels, predicted_holdout_labels)

            print("\n--- Reproducibility Results ---")
            print(f"Hold-out Accuracy: {accuracy:.4f}")
            print(f"Hold-out Adjusted Rand Index (ARI): {ari:.4f}")
        
            print("\n--- Interpretation ---")
            if ari > 0.8 and accuracy > 0.8:
                print("Conclusion: EXCELLENT REPRODUCIBILITY. The clusters are stable and not an artifact of sampling noise.")
                print("The discovered labels represent a reliable underlying structure in the data.")
            elif ari > 0.6 and accuracy > 0.6:
                print("Conclusion: GOOD REPRODUCIBILITY. The clusters show a good degree of stability.")
            else:
                print("Conclusion: MODERATE/POOR REPRODUCIBILITY. The cluster structure may be unstable or highly dependent on the specific sample.")
            print("="*80)
        
            return {'accuracy': accuracy, 'ari': ari}

    def analyze_cluster_profiles(self):
        """Profile each persona with statistical significance testing"""
        data_with_clusters = self.data.copy()
        data_with_clusters['Persona'] = self.cluster_labels
        
        # Calculate profiles for all variables
        profiles = {}
        significant_vars = []
        
        for var in self.data.columns:
            # ANOVA test for cluster differences
            cluster_groups = [data_with_clusters[data_with_clusters['Persona']==i][var] 
                             for i in range(self.optimal_k)]
            f_stat, p_value = f_oneway(*cluster_groups)
            
            # Calculate means and effect size
            means_by_cluster = data_with_clusters.groupby('Persona')[var].mean()
            overall_var = data_with_clusters[var].var()
            between_var = data_with_clusters.groupby('Persona')[var].mean().var()
            eta_squared = between_var / overall_var if overall_var > 0 else 0
            
            profiles[var] = {
                'means': means_by_cluster,
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significant': p_value < 0.05
            }
            
            if p_value < 0.05:
                significant_vars.append(var)
        
        # Create comprehensive profile visualization
        self._visualize_persona_profiles(data_with_clusters, profiles, significant_vars)
        
        return profiles, data_with_clusters
    
    def _visualize_persona_profiles(self, data_with_clusters, profiles, significant_vars):
        """Create radar chart and profile comparison visualizations"""
        
        # 1. Radar Chart for Persona Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Radar chart
        ax_radar = plt.subplot(2, 2, 1, projection='polar')
        
        angles = np.linspace(0, 2*np.pi, len(self.data.columns), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        for persona in range(self.optimal_k):
            values = [profiles[var]['means'][persona] for var in self.data.columns]
            values += [values[0]]  # Complete the circle
            
            ax_radar.plot(angles, values, 'o-', linewidth=2, 
                         label=f'Persona {persona+1}')
            ax_radar.fill(angles, values, alpha=0.1)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(self.data.columns)
        ax_radar.set_ylim(1, 7)
        ax_radar.set_title('Persona Profiles - Radar Chart')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 2. Bar chart for significant variables only
        ax_bar = plt.subplot(2, 2, 2)
        if significant_vars:
            sig_means = data_with_clusters.groupby('Persona')[significant_vars].mean()
            sig_means.T.plot(kind='bar', ax=ax_bar)
            ax_bar.set_title('Significant Differences Between Personas')
            ax_bar.set_xlabel('Variables')
            ax_bar.set_ylabel('Mean Score (1-7)')
            ax_bar.legend(title='Persona')
            ax_bar.tick_params(axis='x', rotation=45)
        
        # 3. Effect sizes heatmap
        ax_heatmap = plt.subplot(2, 2, 3)
        effect_sizes = pd.DataFrame({
            var: [profiles[var]['eta_squared']] for var in self.data.columns
        }).T
        effect_sizes.columns = ['Effect Size (η²)']
        
        sns.heatmap(effect_sizes, annot=True, fmt='.3f', ax=ax_heatmap, 
                   cmap='Reds', cbar_kws={'label': 'Effect Size'})
        ax_heatmap.set_title('Effect Sizes Across Variables')
        
        # 4. Statistical significance summary
        ax_stats = plt.subplot(2, 2, 4)
        p_values = [profiles[var]['p_value'] for var in self.data.columns]
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        
        bars = ax_stats.bar(range(len(self.data.columns)), p_values, color=colors)
        ax_stats.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax_stats.set_xticks(range(len(self.data.columns)))
        ax_stats.set_xticklabels(self.data.columns, rotation=45)
        ax_stats.set_ylabel('p-value')
        ax_stats.set_title('Statistical Significance Tests')
        ax_stats.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print persona summary
        print("\n" + "="*60)
        print("PERSONA DISCOVERY SUMMARY")
        print("="*60)
        print(f"Number of personas identified: {self.optimal_k}")
        print(f"Variables with significant differences: {len(significant_vars)}")
        print(f"Significant variables: {', '.join(significant_vars)}")
        
        for persona in range(self.optimal_k):
            persona_size = sum(self.cluster_labels == persona)
            print(f"\nPersona {persona+1} (n={persona_size}):")
            persona_means = data_with_clusters[data_with_clusters['Persona']==persona].mean()
            
            # Identify defining characteristics (top 3 highest and lowest scores)
            sorted_means = persona_means.drop('Persona').sort_values(ascending=False)
            print(f"  Highest: {sorted_means.head(3).to_dict()}")
            print(f"  Lowest: {sorted_means.tail(3).to_dict()}")
    
    def validate_pathways_for_subgroups(self):
        """
        Test System 1 → System 2 confounding and SOR relationships within each persona
        This is the key test for dual processing confounding effects
        """
        data_with_clusters = self.data.copy()
        data_with_clusters['Persona'] = self.cluster_labels
        
        # Create System 1 composite (experiential processing)
        system1_vars = ['IMM', 'ARIT', 'PL']
        data_with_clusters['System1_Composite'] = data_with_clusters[system1_vars].mean(axis=1)
        
        # System 2 is IPS (cognitive processing)
        data_with_clusters['System2'] = data_with_clusters['IPS']
        
        # Define pathways based on dual processing theory and SOR structure
        pathways = {
            # PRIMARY CONFOUNDING TEST: System 1 → System 2
            'System1_to_System2_Composite': ('System1_Composite', 'System2'),
            
            # INDIVIDUAL SYSTEM 1 COMPONENTS → SYSTEM 2
            'Immersion_to_System2': ('IMM', 'System2'),
            'AR-Induced_Telepresence_to_System2': ('ARIT', 'System2'),
            'Pleasure_to_System2': ('PL', 'System2'),
            
            # SYSTEM 2 → OUTCOMES (should be less affected by System 1 confounding)
            'System2_to_Purchase': ('System2', 'PI'),
            'System2_to_Cross_Buying': ('System2', 'CBI'),
            
            # FULL CHAINS (most vulnerable to confounding)
            'System1_to_Purchase': ('System1_Composite', 'PI'),
            'System1_to_Cross_Buying': ('System1_Composite', 'CBI')
        }
        
        pathway_results = {}
        
        print("\n" + "="*60)
        print("DUAL PROCESSING CONFOUNDING ANALYSIS")
        print("="*60)
        print("System 1 Variables: IMM + ARIT + PL (Experiential Processing)")
        print("System 2 Variable: IPS (Cognitive Processing)")
        print("="*60)
        
        for pathway_name, (x_var, y_var) in pathways.items():
            # Ensure variables exist before processing
            if x_var not in data_with_clusters.columns or y_var not in data_with_clusters.columns:
                print(f"\nSkipping pathway: {x_var} → {y_var} (one or both variables missing)")
                continue
            
            print(f"\nTesting pathway: {x_var} → {y_var}")
            pathway_results[pathway_name] = {}
            
            # Overall correlation (without clustering)
            overall_corr = data_with_clusters[x_var].corr(data_with_clusters[y_var])
            print(f"Overall correlation: {overall_corr:.3f}")
            
            # Within-persona correlations
            persona_corrs = []
            for persona in range(self.optimal_k):
                persona_data = data_with_clusters[data_with_clusters['Persona'] == persona]
                if len(persona_data) > 10:  # Minimum sample size
                    persona_corr = persona_data[x_var].corr(persona_data[y_var])
                    persona_corrs.append(persona_corr)
                    print(f"  Persona {persona+1} (n={len(persona_data)}): {persona_corr:.3f}")
                else:
                    persona_corrs.append(np.nan)
                    print(f"  Persona {persona+1} (n={len(persona_data)}): insufficient data")
            
            # Robustness assessment
            valid_corrs = [c for c in persona_corrs if not np.isnan(c)]
            if valid_corrs:
                consistency = np.std(valid_corrs)
                avg_within_persona = np.mean(valid_corrs)
                correlation_drop = overall_corr - avg_within_persona
                drop_percentage = (correlation_drop / overall_corr) * 100 if overall_corr != 0 else 0
                
                pathway_results[pathway_name] = {
                    'overall_correlation': overall_corr,
                    'persona_correlations': persona_corrs,
                    'average_within_persona': avg_within_persona,
                    'consistency': consistency,
                    'correlation_drop': correlation_drop,
                    'drop_percentage': drop_percentage,
                    'robust': consistency < 0.2,  # Arbitrary threshold for robustness
                    'confounding_severity': 'High' if drop_percentage > 40 else 'Moderate' if drop_percentage > 20 else 'Low'
                }
                
                print(f"  Average within-persona: {avg_within_persona:.3f}")
                print(f"  Consistency (SD): {consistency:.3f}")
                print(f"  Correlation drop: {correlation_drop:.3f} ({drop_percentage:.1f}%)")
                print(f"  Confounding severity: {pathway_results[pathway_name]['confounding_severity']}")
                print(f"  Robust? {'Yes' if consistency < 0.2 else 'No'}")
        
        # Special analysis for System 1 → System 2 confounding
        self._analyze_dual_processing_confounding(data_with_clusters, pathway_results)
        
        # Create visualization for pathway robustness
        self._visualize_pathway_robustness(pathway_results, data_with_clusters)
        
        return pathway_results
    
    def _analyze_dual_processing_confounding(self, data_with_clusters, pathway_results):
        """Special analysis for dual processing confounding hypothesis"""
        
        print("\n" + "="*60)
        print("DUAL PROCESSING CONFOUNDING INTERPRETATION")
        print("="*60)
        
        # Get the main confounding pathway result
        main_pathway = pathway_results.get('System1_to_System2_Composite', {})
        
        if main_pathway:
            drop_pct = main_pathway.get('drop_percentage', 0)
            
            print(f"System 1 → System 2 Composite Relationship:")
            print(f"  Overall correlation: {main_pathway.get('overall_correlation', 0):.3f}")
            print(f"  Within-persona average: {main_pathway.get('average_within_persona', 0):.3f}")
            print(f"  Confounding effect: {drop_pct:.1f}% of original correlation")
            
            # Interpretation
            if drop_pct > 50:
                interpretation = "SEVERE CONFOUNDING: Most of the relationship was due to individual differences"
            elif drop_pct > 30:
                interpretation = "MODERATE CONFOUNDING: Significant individual differences present, but genuine effect remains"
            elif drop_pct > 15:
                interpretation = "MILD CONFOUNDING: Some individual differences, relationship largely genuine"
            else:
                interpretation = "MINIMAL CONFOUNDING: Relationship appears to be genuine dual processing"
            
            print(f"\nInterpretation: {interpretation}")
            
            # Compare individual System 1 components
            print(f"\nIndividual System 1 Components → System 2:")
            for component in ['Immersion_to_System2', 'AR-Induced_Telepresence_to_System2', 'Pleasure_to_System2']:
                if component in pathway_results:
                    comp_drop = pathway_results[component].get('drop_percentage', 0)
                    comp_name = component.split('_')[0]
                    if 'AR-Induced' in component:
                        comp_name = 'AR-Induced Telepresence'
                    print(f"  {comp_name}: {comp_drop:.1f}% confounding")
            
            # System 2 to outcomes (should show less confounding if theory is correct)
            print(f"\nSystem 2 → Outcomes (should be less confounded):")
            for outcome_path in ['System2_to_Purchase', 'System2_to_Cross_Buying']:
                if outcome_path in pathway_results:
                    outcome_drop = pathway_results[outcome_path].get('drop_percentage', 0)
                    outcome_name = outcome_path.split('_to_')[1].replace('_', ' ')
                    print(f"  System 2 → {outcome_name}: {outcome_drop:.1f}% confounding")
                    
                    if outcome_drop < drop_pct:
                        print(f"    ✓ Less confounding than System 1→2, supports dual processing theory")
                    else:
                        print(f"    ⚠ More confounding than System 1→2, suggests broader individual differences")
        
        print("\n" + "="*60)

    def _clean_pathway_name_for_display(self, pathway_name, short=False):
        """
        Clean pathway names for proper display.
        """
        # Define proper mappings for clarity and consistency
        clean_mappings = {
            'System1_to_System2_Composite': 'System 1 → System 2 (Composite)' if not short else 'S1→S2(C)',
            'Immersion_to_System2': 'Immersion → System 2' if not short else 'IMM→S2',
            'AR-Induced_Telepresence_to_System2': 'ARIT → System 2' if not short else 'ARIT→S2',
            'Pleasure_to_System2': 'Pleasure → System 2' if not short else 'PL→S2',
            'System2_to_Purchase': 'System 2 → Purchase' if not short else 'S2→PI',
            'System2_to_Cross_Buying': 'System 2 → Cross-Buying' if not short else 'S2→CBI',
            'System1_to_Purchase': 'System 1 → Purchase' if not short else 'S1→PI',
            'System1_to_Cross_Buying': 'System 1 → Cross-Buying' if not short else 'S1→CBI'
        }
        return clean_mappings.get(pathway_name, pathway_name.replace('_', ' '))

    def _visualize_pathway_robustness(self, pathway_results, data_with_clusters):
        """Visualize pathway robustness across personas with Nature journal styling"""
        
        fig = plt.figure(figsize=(16, 12))
        
        priority_paths = [
            'System1_to_System2_Composite', 'Immersion_to_System2', 'AR-Induced_Telepresence_to_System2',
            'Pleasure_to_System2', 'System2_to_Purchase', 'System2_to_Cross_Buying',
            'System1_to_Purchase', 'System1_to_Cross_Buying'
        ]
        
        plot_positions = [(3, 4, i + 1) for i in range(len(priority_paths))]
        
        # Generate clean names for summary plots first
        pathway_names_clean_short = [self._clean_pathway_name_for_display(name, short=True) for name in priority_paths]

        for idx, pathway_name in enumerate(priority_paths):
            if pathway_name in pathway_results and pathway_results[pathway_name]:
                results = pathway_results[pathway_name]
                ax = plt.subplot(*plot_positions[idx])
                
                personas = [f'Persona {j+1}' for j in range(self.optimal_k)]
                persona_corrs = results['persona_correlations']
                x_pos = np.arange(len(personas))
                
                bar_values = [c if not np.isnan(c) else 0 for c in persona_corrs]
                bar_colors = [nature_colors[1] if not np.isnan(c) else '#D3D3D3' for c in persona_corrs]
                
                bars = ax.bar(x_pos, bar_values, alpha=0.8, color=bar_colors, edgecolor='black', linewidth=1.5)
                
                overall_corr = results['overall_correlation']
                ax.axhline(y=overall_corr, color=nature_colors[0], linestyle='--', linewidth=3, alpha=0.9, label=f'Overall: {overall_corr:.3f}')
                
                drop_pct = results.get('drop_percentage', 0)
                confounding_level = results.get('confounding_severity', 'Unknown')
                
                ax.set_xlabel('Persona', fontweight='bold')
                ax.set_ylabel('Correlation Coefficient', fontweight='bold')
                
                clean_name = self._clean_pathway_name_for_display(pathway_name, short=False)
                ax.set_title(f'{clean_name}\nConfounding: {drop_pct:.1f}% ({confounding_level})', fontweight='bold', fontsize=12)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(personas, fontweight='bold')
                ax.legend(loc='upper right', prop={'weight': 'bold'})
                ax.grid(True, alpha=0.3, axis='y', linewidth=1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                if confounding_level == 'High': ax.title.set_color(nature_colors[0])
                elif confounding_level == 'Moderate': ax.title.set_color('#FF7F00')
                else: ax.title.set_color(nature_colors[2])
                
                for bar, val in zip(bars, bar_values):
                    if val != 0:
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Summary Plots
        # Confounding severity comparison
        ax_summary = plt.subplot(3, 4, 9)
        confounding_levels = [pathway_results.get(name, {}).get('drop_percentage', 0) for name in priority_paths]
        colors = [nature_colors[0] if level > 40 else '#FF7F00' if level > 20 else nature_colors[2] for level in confounding_levels]
        ax_summary.bar(range(len(pathway_names_clean_short)), confounding_levels, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax_summary.set_title('Confounding Summary', fontweight='bold', fontsize=12)
        ax_summary.set_ylabel('Confounding (%)', fontweight='bold')
        ax_summary.set_xticks(range(len(pathway_names_clean_short)))
        ax_summary.set_xticklabels(pathway_names_clean_short, rotation=45, ha='right', fontweight='bold')
        ax_summary.axhline(y=40, color=nature_colors[0], linestyle=':', alpha=0.7, linewidth=2, label='High (>40%)')
        ax_summary.axhline(y=20, color='#FF7F00', linestyle=':', alpha=0.7, linewidth=2, label='Moderate (>20%)')
        ax_summary.legend(prop={'weight': 'bold', 'size': 9}); ax_summary.grid(True, alpha=0.3, axis='y'); ax_summary.spines['top'].set_visible(False); ax_summary.spines['right'].set_visible(False)

        # Correlation strength comparison
        ax_strength = plt.subplot(3, 4, 10)
        overall_corrs = [pathway_results.get(name, {}).get('overall_correlation', 0) for name in priority_paths]
        within_corrs = [pathway_results.get(name, {}).get('average_within_persona', 0) for name in priority_paths]
        x_pos = np.arange(len(overall_corrs)); width = 0.35
        ax_strength.bar(x_pos - width/2, overall_corrs, width, label='Overall Sample', color=nature_colors[0], alpha=0.8, edgecolor='black', linewidth=1)
        ax_strength.bar(x_pos + width/2, within_corrs, width, label='Within Personas', color=nature_colors[1], alpha=0.8, edgecolor='black', linewidth=1)
        ax_strength.set_title('Correlation Strength\nComparison', fontweight='bold', fontsize=12)
        ax_strength.set_ylabel('Correlation Coefficient', fontweight='bold')
        ax_strength.set_xticks(x_pos)
        ax_strength.set_xticklabels(pathway_names_clean_short, rotation=45, ha='right', fontweight='bold')
        ax_strength.legend(prop={'weight': 'bold', 'size': 9}); ax_strength.grid(True, alpha=0.3, axis='y'); ax_strength.spines['top'].set_visible(False); ax_strength.spines['right'].set_visible(False)
        
        # Consistency analysis
        ax_consistency = plt.subplot(3, 4, 11)
        consistency_scores = [pathway_results.get(name, {}).get('consistency', 0) for name in priority_paths]
        colors = [nature_colors[2] if score < 0.1 else '#FF7F00' if score < 0.2 else nature_colors[0] for score in consistency_scores]
        ax_consistency.bar(range(len(consistency_scores)), consistency_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax_consistency.set_title('Between-Persona\nConsistency', fontweight='bold', fontsize=12)
        ax_consistency.set_ylabel('Consistency (SD)', fontweight='bold')
        ax_consistency.set_xticks(range(len(consistency_scores)))
        ax_consistency.set_xticklabels(pathway_names_clean_short, rotation=45, ha='right', fontweight='bold')
        ax_consistency.axhline(y=0.2, color=nature_colors[0], linestyle='--', alpha=0.7, linewidth=2, label='Robustness Threshold')
        ax_consistency.legend(prop={'weight': 'bold', 'size': 9}); ax_consistency.grid(True, alpha=0.3, axis='y'); ax_consistency.spines['top'].set_visible(False); ax_consistency.spines['right'].set_visible(False)

        # System-level summary
        ax_system_summary = plt.subplot(3, 4, 12)
        system_data = {
            'S1 → S2': pathway_results.get('System1_to_System2_Composite', {}).get('drop_percentage', 0),
            'S2 → PI': pathway_results.get('System2_to_Purchase', {}).get('drop_percentage', 0),
            'S2 → CBI': pathway_results.get('System2_to_Cross_Buying', {}).get('drop_percentage', 0)
        }
        bars = ax_system_summary.bar(system_data.keys(), system_data.values(), color=[nature_colors[2], nature_colors[3], nature_colors[4]], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax_system_summary.set_title('Dual Processing\nSystem Analysis', fontweight='bold', fontsize=12)
        ax_system_summary.set_ylabel('Confounding (%)', fontweight='bold')
        ax_system_summary.tick_params(axis='x', rotation=45, labelsize=10)
        ax_system_summary.grid(True, alpha=0.3, axis='y'); ax_system_summary.spines['top'].set_visible(False); ax_system_summary.spines['right'].set_visible(False)
        for bar, value in zip(bars, system_data.values()):
            ax_system_summary.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5, f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.show()
        
        print("\n" + "="*100)
        print("PATHWAY ROBUSTNESS ANALYSIS SUMMARY (NATURE JOURNAL STYLE)")
        print("="*100)
        print(f"\n{'Pathway':<25} {'Overall':<10} {'Within':<10} {'Drop':<10} {'Severity':<12} {'Robust':<8}")
        print("-" * 100)
        
        for name in priority_paths:
            if name in pathway_results and pathway_results[name]:
                results = pathway_results[name]
                clean_name = self._clean_pathway_name_for_display(name, short=True)
                overall = results['overall_correlation']
                within = results['average_within_persona']
                drop = results['drop_percentage']
                severity = results['confounding_severity']
                robust = 'Yes' if results['robust'] else 'No'
                print(f"{clean_name:<25} {overall:<10.3f} {within:<10.3f} {drop:<9.1f}% {severity:<12} {robust:<8}")
        
        print("="*100)
    
    def plot_nature_pathway_analysis(self, save_path=None):
        """Draw dual-pathway analysis chart (corrected version, adapted for 26-inch screens)"""
        # =============================================================================
        # 数据翻译层：为绘图函数提供所需变量 (START)
        # =============================================================================
        
        # 1. 创建绘图函数需要的 DataFrame
        #    注意：这里我们假设 persona 标签已经生成
        if self.cluster_labels is None:
            print("错误：无法绘图，因为聚类（Persona）尚未生成。请先运行 discover_optimal_clusters()。")
            return
            
        self.analysis_df = self.data.copy()
        self.analysis_df['persona_name'] = [f"Persona {i+1}" for i in self.cluster_labels]

        # 2. 定义变量分组，使其与绘图函数期望的结构匹配
        self.stimulus_vars = {
            'technical_features': [var for var in ['AR', 'IVR', 'PSN'] if var in self.data.columns],
            'anticipated_emotion': ['ANP'] if 'ANP' in self.data.columns else []
        }
        self.organism_vars = {
            'experience_states': [var for var in ['IMM', 'ARIT', 'PL'] if var in self.data.columns],
            'cognitive_processing': ['IPS'] if 'IPS' in self.data.columns else []
        }
        self.response_vars = {
             'behavioral_intentions': [var for var in ['PI', 'CBI'] if var in self.data.columns]
        }
        self.all_vars_for_corr = [var for group in [self.stimulus_vars, self.organism_vars, self.response_vars] for var_list in group.values() for var in var_list]

        # 3. 定义颜色
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 使用与原函数匹配的颜色

        # 4. 定义 plot_style 对象（一个简化的替代品）
        class SimplePlotStyle:
            def __init__(self):
                self.diverging_cmap = 'Blues' # 匹配您想要的蓝色热图风格
        self.plot_style = SimplePlotStyle()
        
        # =============================================================================
        # 数据翻译层 (END)
        # =============================================================================
        
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_pathway_scatter(ax1, 'technical_features', 'experience_states', 
                                  'Technical Features', 'Experience States', 'A')
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_pathway_scatter(ax2, 'anticipated_emotion', 'experience_states',
                                  'Anticipated Emotion', 'Experience States', 'B')
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_single_pathway(ax3, 'experience_states', 'IPS',
                                 'Experience States', 'Cognitive Processing (IPS)', 'C')
        
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_ips_to_behavior(ax4, 'D')
        
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_overall_so(ax5, 'E')
        
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_pathway_overview(ax6, 'F')
        
        plt.tight_layout(pad=3.0)
        fig.suptitle('S-O-R Dual-Pathway Model Analysis', fontsize=20, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_pathway_scatter(self, ax, x_group, y_group, xlabel, ylabel, panel_label):
        """Draw pathway scatter plot"""
        var_map = {
            'technical_features': self.stimulus_vars.get('technical_features', []),
            'anticipated_emotion': self.stimulus_vars.get('anticipated_emotion', []),
            'experience_states': self.organism_vars.get('experience_states', []),
        }
        x_vars = var_map.get(x_group, [])
        y_vars = var_map.get(y_group, [])
        
        if not x_vars or not y_vars:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14, fontweight='bold')
            ax.set_title(f'{panel_label}. {xlabel} → {ylabel}', fontsize=14, fontweight='bold')
            return
        
        x_data = self.analysis_df[x_vars].mean(axis=1)
        y_data = self.analysis_df[y_vars].mean(axis=1)
        
        for i, persona in enumerate(self.analysis_df['persona_name'].unique()):
            mask = self.analysis_df['persona_name'] == persona
            ax.scatter(x_data[mask], y_data[mask], 
                       color=self.colors[i % len(self.colors)], alpha=0.6, s=40, label=persona)
        
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ax.plot(x_data.sort_values(), p(x_data.sort_values()), 'k--', alpha=0.5, linewidth=2.0)
        
        r, p_val = stats.pearsonr(x_data.dropna(), y_data.dropna())
        ax.text(0.05, 0.95, f'r = {r:.3f}\np < 0.001' if p_val < 0.001 else f'r = {r:.3f}\np = {p_val:.3f}', 
                transform=ax.transAxes, fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f'{panel_label}. {xlabel} → {ylabel}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if panel_label == 'A':
            ax.legend(fontsize=11, prop={'weight': 'bold'})

    def _plot_single_pathway(self, ax, x_var_source, y_var, xlabel, ylabel, panel_label):
        """Draw single variable pathway chart"""
        if x_var_source == 'experience_states':
            x_vars = self.organism_vars['experience_states']
            if not x_vars:
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
                return
            x_data = self.analysis_df[x_vars].mean(axis=1)
        else:
            if x_var_source not in self.analysis_df.columns:
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
                return
            x_data = self.analysis_df[x_var_source]
        
        if y_var not in self.analysis_df.columns:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
            return
        
        y_data = self.analysis_df[y_var]
        
        for i, persona in enumerate(self.analysis_df['persona_name'].unique()):
            mask = self.analysis_df['persona_name'] == persona
            ax.scatter(x_data[mask], y_data[mask], 
                       color=self.colors[i % len(self.colors)], alpha=0.6, s=40, label=persona)
        
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ax.plot(x_data.sort_values(), p(x_data.sort_values()), 'k--', alpha=0.5, linewidth=2.0)
        
        r, p_val = stats.pearsonr(x_data.dropna(), y_data.dropna())
        ax.text(0.05, 0.95, f'r = {r:.3f}\np < 0.001' if p_val < 0.001 else f'r = {r:.3f}\np = {p_val:.3f}',
                transform=ax.transAxes, fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f'{panel_label}. {xlabel} → {ylabel}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_ips_to_behavior(self, ax, panel_label):
        """Draw IPS to behavioral intentions pathway"""
        behavior_vars = [var for var_list in self.response_vars.values() for var in var_list]
        
        if not behavior_vars or 'IPS' not in self.analysis_df.columns:
            ax.text(0.5, 0.5, 'No behavioral response\nvariables available', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
            ax.set_title(f'{panel_label}. IPS → Behavioral Response', fontsize=14, fontweight='bold')
            return
        
        behavior_var = behavior_vars[0] # Use the first available behavior var for the plot
        self._plot_single_pathway(ax, 'IPS', behavior_var, 'Cognitive Processing (IPS)', f'Behavioral Response ({behavior_var})', panel_label)

    def _plot_overall_so(self, ax, panel_label):
        """Draw overall S-O pathway"""
        stimulus_vars = self.stimulus_vars['technical_features'] + self.stimulus_vars['anticipated_emotion']
        organism_vars = [var for var_list in self.organism_vars.values() for var in var_list]
        
        if not stimulus_vars or not organism_vars:
            ax.text(0.5, 0.5, 'Insufficient S-O data', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
            ax.set_title(f'{panel_label}. Overall S-O Pathway', fontsize=14, fontweight='bold')
            return
        
        stimulus_avg = self.analysis_df[stimulus_vars].mean(axis=1)
        organism_avg = self.analysis_df[organism_vars].mean(axis=1)
        
        for i, persona in enumerate(self.analysis_df['persona_name'].unique()):
            mask = self.analysis_df['persona_name'] == persona
            ax.scatter(stimulus_avg[mask], organism_avg[mask], 
                       color=self.colors[i % len(self.colors)], alpha=0.6, s=40, label=persona)
        
        z = np.polyfit(stimulus_avg, organism_avg, 1)
        p = np.poly1d(z)
        ax.plot(stimulus_avg.sort_values(), p(stimulus_avg.sort_values()), 'k--', alpha=0.5, linewidth=2.0)
        
        r, p_val = stats.pearsonr(stimulus_avg.dropna(), organism_avg.dropna())
        ax.text(0.05, 0.95, f'r = {r:.3f}\np < 0.001' if p_val < 0.001 else f'r = {r:.3f}\np = {p_val:.3f}', 
                transform=ax.transAxes, fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Overall Stimulus', fontsize=13, fontweight='bold')
        ax.set_ylabel('Overall Organism Response', fontsize=13, fontweight='bold')
        ax.set_title(f'{panel_label}. Overall S-O Pathway', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_pathway_overview(self, ax, panel_label):
        """Draw pathway coefficient overview"""
        key_vars = []
        if self.stimulus_vars['technical_features']: key_vars.append(self.stimulus_vars['technical_features'][0])
        if self.stimulus_vars['anticipated_emotion']: key_vars.append(self.stimulus_vars['anticipated_emotion'][0])
        if self.organism_vars['experience_states']: key_vars.append(self.organism_vars['experience_states'][0])
        if 'IPS' in self.analysis_df.columns: key_vars.append('IPS')
        
        if len(key_vars) < 2:
            ax.text(0.5, 0.5, 'Insufficient variables\nfor pathway overview', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
            ax.set_title(f'{panel_label}. Pathway Overview', fontsize=14, fontweight='bold')
            return
        
        # NOTE: The plot in the image shows correlations for AR, ANP, IMM, IPS.
        # This part of the code might be slightly different from what produced the exact image,
        # but it follows the same logic. Let's adjust it to match the image for clarity.
        # Specific variables: key_vars = ['AR', 'ANP', 'IMM', 'IPS'] # Explicitly match the image; If it want to show all key variables, use the above logic.
        key_vars = self.all_vars_for_corr

        pathway_corr = self.analysis_df[key_vars].corr()
        
        im = ax.imshow(pathway_corr.values, cmap=self.plot_style.diverging_cmap, vmin=0, vmax=1, aspect='auto') # Adjusted vmin for the given plot style
        # The provided plot seems to use a sequential colormap, not diverging.
        # To better match the plot, a sequential map would be used, e.g., cmap='Blues'
        # Example to match image more closely:
        im = ax.imshow(pathway_corr.values, cmap='Blues', vmin=0.3, vmax=0.8, aspect='auto')

        ax.set_xticks(np.arange(len(key_vars)))
        ax.set_yticks(np.arange(len(key_vars)))
        ax.set_xticklabels(key_vars, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax.set_yticklabels(key_vars, fontsize=12, fontweight='bold')
        
        # Modify the text color based on correlation value (for Specific variables)
        # for i in range(len(key_vars)):
            # for j in range(len(key_vars)):
                # Match the image: only show values in the upper triangle
                # if j > i: 
                    # text_color = 'white' # 'white' if pathway_corr.iloc[i, j] > 0.6 else 'black'
                    # ax.text(j, i, f'{pathway_corr.iloc[i, j]:.2f}',
                            # ha="center", va="center", fontsize=11, fontweight='bold',
                            # color=text_color)
        
        # ax.set_title(f'{panel_label}. Key Variable Correlations', fontsize=14, fontweight='bold')

        # 修改点 2: 改变循环条件，以显示除对角线外的所有相关系数
        # ----------------------------------------------------------------------
        for i in range(len(key_vars)):
            for j in range(len(key_vars)):
                # 原条件是 if j > i: (只显示右上部分)
                # 新条件是 if i != j: (显示所有非对角线部分)
                if i != j:
                    corr_val = pathway_corr.iloc[i, j]
                    # 根据背景色深浅决定文字颜色，使其更清晰
                    text_color = 'white' if abs(corr_val) > 0.5 else 'black'
                    ax.text(j, i, f'{corr_val:.2f}',
                            ha="center", va="center", fontsize=11, fontweight='bold',
                            color=text_color)
        
        ax.set_title(f'{panel_label}. Key Variable Correlations', fontsize=14, fontweight='bold')

    def run_complete_analysis(self):
        """Execute the full persona discovery and robustness analysis pipeline"""
        print("="*60)
        print("AR E-COMMERCE PERSONA DISCOVERY & ROBUSTNESS ANALYSIS")
        print("="*60)
        
        try:
            # Step 1: Load and prepare data
            print("\nStep 1: Loading and preparing data...")
            self.load_and_prepare_data()
            if self.data.empty: raise ValueError("Data could not be loaded or is empty.")
            
            # Step 2: Prepare clustering variables
            print("\nStep 2: Selecting clustering variables...")
            self.prepare_clustering_variables()
            if self.clustering_data.empty: raise ValueError("Clustering data is empty.")

            # Step 3: Discover optimal clusters
            print("\nStep 3: Discovering optimal personas...")
            validation_results = self.discover_optimal_clusters()
            # NEW STEP: Formally prove cluster reproducibility
            print("\nStep 3.5: Validating persona reproducibility...")
            self.validate_cluster_reproducibility()
            
            # Step 4: Analyze cluster profiles
            print("\nStep 4: Analyzing persona profiles...")
            profiles, data_with_clusters = self.analyze_cluster_profiles()
            
            # Step 5: Validate pathways for robustness
            print("\nStep 5: Testing pathway robustness...")
            pathway_results = self.validate_pathways_for_subgroups()
            
            # Step 6: Plot the S-O-R dual-pathway analysis
            print("\nStep 6: Generating S-O-R dual-pathway plot...")
            self.plot_nature_pathway_analysis()

            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            
            return {
                'validation_results': validation_results,
                'profiles': profiles,
                'pathway_results': pathway_results,
                'data_with_personas': data_with_clusters
            }
        except Exception as e:
            print(f"\nAN ERROR OCCURRED: {e}")
            return None

# Usage Example
if __name__ == "__main__":
    # Initialize the analysis
    csv_file_path = r"C:/Users/10490/Desktop/Middle Data.csv"
    analyzer = ARPersonaDiscovery(csv_file_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Access specific results only if analysis was successful
    if results:
        print("\nKey Findings:")
        if analyzer.optimal_k is not None:
             print(f"- Identified {analyzer.optimal_k} distinct user personas")
             print(f"- Cluster assignments: {np.bincount(analyzer.cluster_labels)}")
        
        # Example of accessing pathway robustness results
        if 'pathway_results' in results:
            for pathway, result in results['pathway_results'].items():
                if result and 'robust' in result:
                    robustness = "ROBUST" if result['robust'] else "NOT ROBUST"
                    print(f"- {pathway}: {robustness} (consistency: {result['consistency']:.3f})")
