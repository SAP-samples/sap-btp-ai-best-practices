import { apiService } from '../../services/api.js';
import { pageRouter } from '../../modules/router.js';

class PDFExtractionPage {
  constructor() {
    this.selectedFile = null;
    this.extractedData = null;
    this.processingStartTime = null;
    
    this.initializeElements();
    this.attachEventListeners();
    
    // Debug: Log initialization
    console.log('PDFExtractionPage initialized');
    console.log('PDF Uploader element:', this.pdfUploader);
    console.log('Extract button element:', this.extractButton);
  }

  initializeElements() {
    // Get references to UI elements
    this.pdfUploader = document.getElementById('pdf-uploader');
    this.extractButton = document.getElementById('extract-button');
    this.progressIndicator = document.getElementById('progress-indicator');
    this.statusMessage = document.getElementById('status-message');
    
    // Result cards
    this.headerCard = document.getElementById('header-card');
    this.tableCard = document.getElementById('table-card');
    this.detailsCard = document.getElementById('details-card');
    
    // Header info elements
    this.clientText = document.getElementById('client-text');
    this.dateText = document.getElementById('date-text');
    
    // Table and export
    this.lineItemsTable = document.getElementById('line-items-table');
    this.exportButton = document.getElementById('export-button');
    this.createSalesOrderButton = document.getElementById('create-sales-order-button');
    
    // Processing details
    this.modelUsedText = document.getElementById('model-used-text');
    this.tokensUsedText = document.getElementById('tokens-used-text');
    this.processingTimeText = document.getElementById('processing-time-text');
  }

  attachEventListeners() {
    console.log('Attaching event listeners...');
    
    // Wait for UI5 components to be fully loaded
    this.waitForUI5Components().then(() => {
      this.setupEventListeners();
    });
  }

  async waitForUI5Components() {
    // Wait for UI5 components to be defined
    const maxAttempts = 50;
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      if (this.extractButton && this.pdfUploader) {
        // Check if UI5 components are ready
        if (customElements.get('ui5-button') && customElements.get('ui5-file-uploader')) {
          console.log('UI5 components are ready');
          return;
        }
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }
    
    console.warn('UI5 components may not be fully loaded, proceeding anyway');
  }

  setupEventListeners() {
    console.log('Setting up event listeners after UI5 components are ready...');
    
    // File uploader change event
    if (this.pdfUploader) {
      // Try multiple event types for file uploader
      ['change', 'ui5-change', 'input'].forEach(eventType => {
        this.pdfUploader.addEventListener(eventType, (event) => {
          console.log(`File uploader ${eventType} event triggered`);
          this.handleFileSelection(event);
        });
      });
    } else {
      console.error('PDF uploader element not found!');
    }

    // Extract button - try multiple approaches
    if (this.extractButton) {
      console.log('Setting up extract button listeners...');
      
      // Method 1: Standard click event
      this.extractButton.addEventListener('click', (event) => {
        console.log('Extract button click triggered!', event);
        event.preventDefault();
        event.stopPropagation();
        this.handleExtraction();
      });

      // Method 2: Mouse events
      this.extractButton.addEventListener('mousedown', (event) => {
        console.log('Extract button mousedown triggered!', event);
      });

      this.extractButton.addEventListener('mouseup', (event) => {
        console.log('Extract button mouseup triggered!', event);
        if (event.button === 0) { // Left click
          this.handleExtraction();
        }
      });

      // Method 3: Touch events for mobile
      this.extractButton.addEventListener('touchstart', (event) => {
        console.log('Extract button touchstart triggered!', event);
      });

      this.extractButton.addEventListener('touchend', (event) => {
        console.log('Extract button touchend triggered!', event);
        event.preventDefault();
        this.handleExtraction();
      });

      // Method 4: Direct onclick property
      this.extractButton.onclick = (event) => {
        console.log('Extract button onclick property triggered!', event);
        this.handleExtraction();
      };

      // Method 5: Add tabindex and keyboard support
      this.extractButton.setAttribute('tabindex', '0');
      this.extractButton.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          console.log('Extract button keyboard triggered!', event);
          event.preventDefault();
          this.handleExtraction();
        }
      });

    } else {
      console.error('Extract button element not found!');
    }

    // Export button
    if (this.exportButton) {
      this.exportButton.addEventListener('click', (event) => {
        console.log('Export button click triggered');
        event.preventDefault();
        this.handleExport();
      });
    }

    // Create Sales Order button
    if (this.createSalesOrderButton) {
      this.createSalesOrderButton.addEventListener('click', (event) => {
        console.log('Create Sales Order button click triggered');
        event.preventDefault();
        this.handleCreateSalesOrder();
      });
    }

    // Debug button
    const debugButton = document.getElementById('debug-enable-button');
    if (debugButton) {
      debugButton.addEventListener('click', (event) => {
        console.log('Debug button click triggered');
        event.preventDefault();
        // Create a mock file for testing
        const mockFile = new File(['mock pdf content'], 'test.pdf', { type: 'application/pdf' });
        this.selectedFile = mockFile;
        this.extractButton.disabled = false;
        this.showMessage('Debug: Extract button enabled with mock file', 'Information');
      });
    }
    
    console.log('All event listeners attached successfully');
  }

  handleFileSelection(event) {
    console.log('File selection event:', event);
    
    // Try different ways to get files from UI5 file uploader
    let files = null;
    
    // Method 1: Standard files property
    if (event.target.files) {
      files = event.target.files;
    }
    // Method 2: UI5 specific detail property
    else if (event.detail && event.detail.files) {
      files = event.detail.files;
    }
    // Method 3: Direct access to uploader files
    else if (this.pdfUploader.files) {
      files = this.pdfUploader.files;
    }
    
    console.log('Files found:', files);
    
    if (files && files.length > 0) {
      const file = files[0];
      console.log('Selected file:', file);
      
      // Validate file type - now supports PDF, Excel, and CSV
      const fileExtension = file.name.toLowerCase().split('.').pop();
      const supportedTypes = ['pdf', 'xlsx', 'xls', 'csv'];
      
      if (!supportedTypes.includes(fileExtension)) {
        this.showMessage('Please select a PDF, Excel (.xlsx, .xls), or CSV file.', 'Error');
        this.extractButton.disabled = true;
        return;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        this.showMessage('File size must be less than 10MB.', 'Error');
        this.extractButton.disabled = true;
        return;
      }

      this.selectedFile = file;
      this.extractButton.disabled = false;
      this.hideResultCards();
      
      // Show appropriate message based on file type
      const fileType = fileExtension === 'pdf' ? 'PDF' : 
                      ['xlsx', 'xls'].includes(fileExtension) ? 'Excel' : 'CSV';
      this.showMessage(`Selected: ${file.name} (${fileType}, ${this.formatFileSize(file.size)})`, 'Information');
    } else {
      this.selectedFile = null;
      this.extractButton.disabled = true;
      this.hideMessage();
    }
  }

  async handleExtraction() {
    if (!this.selectedFile) {
      this.showMessage('Please select a document file first.', 'Error');
      return;
    }

    try {
      this.setLoadingState(true);
      this.processingStartTime = Date.now();
      
      // Detect file type
      const fileExtension = this.selectedFile.name.toLowerCase().split('.').pop();
      const isPDF = fileExtension === 'pdf';
      const isExcel = ['xlsx', 'xls', 'csv'].includes(fileExtension);
      
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', this.selectedFile);
      
      // Call appropriate API endpoint based on file type
      let response;
      if (isPDF) {
        response = await apiService.uploadPDF(formData);
      } else if (isExcel) {
        response = await apiService.uploadExcel(formData);
      } else {
        throw new Error('Unsupported file type');
      }
      
      if (response.success) {
        this.extractedData = response;
        this.displayResults(response);
        
        const fileType = isPDF ? 'PDF' : 'Excel/CSV';
        this.showMessage(`${fileType} data extracted successfully!`, 'Success');
      } else {
        throw new Error(response.error || 'Extraction failed');
      }
      
    } catch (error) {
      console.error('Extraction error:', error);
      this.showMessage(`Extraction failed: ${error.message}`, 'Error');
      this.hideResultCards();
    } finally {
      this.setLoadingState(false);
    }
  }

  displayResults(data) {
    console.log('Displaying results:', data);
    console.log('Extracted data:', data.extracted_data);
    
    // Display header information - support both old and new formats
    const header = data.extracted_data?.header || {};
    const clientValue = header.customer || header.client || 'Not found';
    const dateValue = header.date || 'Not found';
    
    this.clientText.textContent = clientValue;
    this.dateText.textContent = dateValue;
    
    // Display line items in table - support both old and new formats
    const lineItems = data.extracted_data?.positions || data.extracted_data?.line_items || [];
    console.log('Line items to display:', lineItems);
    this.populateTable(lineItems);
    
    // Display processing details
    const processingTime = this.processingStartTime ? 
      ((Date.now() - this.processingStartTime) / 1000).toFixed(2) + 's' : 'Unknown';
    
    this.modelUsedText.textContent = data.model_used || 'Unknown';
    this.tokensUsedText.textContent = this.formatTokenUsage(data.usage);
    this.processingTimeText.textContent = processingTime;
    
    // Show result cards with animation
    this.showResultCards();
  }

  populateTable(lineItems) {
    // Clear existing rows (except header)
    const existingRows = this.lineItemsTable.querySelectorAll('ui5-table-row');
    existingRows.forEach(row => row.remove());
    
    // Add new rows
    lineItems.forEach((item, index) => {
      const row = document.createElement('ui5-table-row');
      
      // Material cell
      const materialCell = document.createElement('ui5-table-cell');
      materialCell.textContent = item.material || '-';
      row.appendChild(materialCell);
      
      // Quantity cell
      const quantityCell = document.createElement('ui5-table-cell');
      quantityCell.textContent = item.quantity || '-';
      row.appendChild(quantityCell);
      
      // Unit Price cell - support both old and new formats
      const priceCell = document.createElement('ui5-table-cell');
      const priceValue = item.price || item.unit_price || '-';
      priceCell.textContent = priceValue;
      row.appendChild(priceCell);
      
      this.lineItemsTable.appendChild(row);
    });
  }

  handleExport() {
    if (!this.extractedData || !this.extractedData.extracted_data) {
      this.showMessage('No data to export.', 'Warning');
      return;
    }

    try {
      const csvContent = this.generateCSV(this.extractedData.extracted_data);
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      
      if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `extracted_data_${Date.now()}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        this.showMessage('Data exported successfully!', 'Success');
      }
    } catch (error) {
      console.error('Export error:', error);
      this.showMessage('Export failed. Please try again.', 'Error');
    }
  }

  generateCSV(data) {
    let csv = 'Type,Field,Value\n';
    
    // Add header data - support both old and new formats
    const header = data.header || {};
    const clientValue = header.customer || header.client || '';
    csv += `Header,Client,"${clientValue}"\n`;
    csv += `Header,Date,"${header.date || ''}"\n`;
    
    // Add line items - support both old and new formats
    const lineItems = data.positions || data.line_items || [];
    if (lineItems.length > 0) {
      csv += '\nMaterial,Quantity,Unit Price\n';
      lineItems.forEach(item => {
        const priceValue = item.price || item.unit_price || '';
        csv += `"${item.material || ''}","${item.quantity || ''}","${priceValue}"\n`;
      });
    }
    
    return csv;
  }

  setLoadingState(loading) {
    if (loading) {
      this.extractButton.disabled = true;
      this.extractButton.loading = true;
      this.extractButton.textContent = 'Processing...';
      this.progressIndicator.style.display = 'block';
      this.progressIndicator.value = 50; // Indeterminate progress
      document.querySelector('.upload-section').classList.add('loading');
    } else {
      this.extractButton.disabled = false;
      this.extractButton.loading = false;
      this.extractButton.textContent = 'Process with GenAI';
      this.progressIndicator.style.display = 'none';
      document.querySelector('.upload-section').classList.remove('loading');
    }
  }

  showMessage(message, type) {
    this.statusMessage.textContent = message;
    this.statusMessage.design = type;
    this.statusMessage.style.display = 'block';
    
    // Auto-hide success messages after 5 seconds
    if (type === 'Success') {
      setTimeout(() => {
        this.hideMessage();
      }, 5000);
    }
  }

  hideMessage() {
    this.statusMessage.style.display = 'none';
  }

  showResultCards() {
    this.headerCard.style.display = 'block';
    this.tableCard.style.display = 'block';
    this.detailsCard.style.display = 'block';
    
    // Add fade-in animation
    setTimeout(() => {
      this.headerCard.classList.add('card-fade-in');
      this.tableCard.classList.add('card-fade-in');
      this.detailsCard.classList.add('card-fade-in');
    }, 100);
  }

  hideResultCards() {
    this.headerCard.style.display = 'none';
    this.tableCard.style.display = 'none';
    this.detailsCard.style.display = 'none';
    
    // Remove animation classes
    this.headerCard.classList.remove('card-fade-in');
    this.tableCard.classList.remove('card-fade-in');
    this.detailsCard.classList.remove('card-fade-in');
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  formatTokenUsage(usage) {
    if (!usage) return 'Unknown';
    
    if (usage.total_tokens) {
      return `${usage.total_tokens} tokens`;
    } else if (usage.inputTokens && usage.outputTokens) {
      return `${usage.inputTokens + usage.outputTokens} tokens`;
    } else if (usage.prompt_tokens && usage.completion_tokens) {
      return `${usage.prompt_tokens + usage.completion_tokens} tokens`;
    }
    
    return 'Unknown';
  }

  handleCreateSalesOrder() {
    if (!this.extractedData || !this.extractedData.extracted_data) {
      this.showMessage('No data available to create Sales Order.', 'Warning');
      return;
    }

    try {
      // Get extracted line items
      const lineItems = this.extractedData.extracted_data?.positions || this.extractedData.extracted_data?.line_items || [];
      
      if (lineItems.length === 0) {
        this.showMessage('No line items found to create Sales Order.', 'Warning');
        return;
      }

      // Transform data for Sales Order with random materials and values
      const salesOrderItems = lineItems
        .filter(item => {
          // Only include items with quantity >= 1
          const quantity = parseFloat(item.quantity) || 0;
          return quantity >= 1;
        })
        .map((item, index) => {
          const material = this.generateRandomMaterial();
          const netValue = this.generateRandomNetValue();
          const currentDate = new Date();
          const currentTime = currentDate.toLocaleTimeString('en-US');
          
          return {
            item: (index + 1) * 10, // 10, 20, 30, etc.
            material: material,
            description: item.material || item.description || `Product ${index + 1}`,
            quantity: item.quantity || '1.000',
            unit: 'PC',
            netValue: `${netValue} USD`,
            createdBy: 'SAP AI',
            createdOn: currentDate.toLocaleDateString('en-US', { 
              year: 'numeric', 
              month: 'short', 
              day: 'numeric' 
            }),
            createdAt: currentTime
          };
        });

      // Store data in sessionStorage for the Sales Order page
      const salesOrderData = {
        items: salesOrderItems,
        sourceData: this.extractedData.extracted_data
      };
      
      sessionStorage.setItem('salesOrderData', JSON.stringify(salesOrderData));
      
      // Navigate to Sales Order page using the router
      pageRouter.navigate('/sales-order');
      
      this.showMessage('Navigating to Sales Order...', 'Information');
      
    } catch (error) {
      console.error('Create Sales Order error:', error);
      this.showMessage('Failed to create Sales Order. Please try again.', 'Error');
    }
  }

  generateRandomMaterial() {
    // Generate a 6-digit random material number
    // Ensure it doesn't repeat in the same session
    if (!window.usedMaterials) {
      window.usedMaterials = new Set();
    }
    
    let material;
    let attempts = 0;
    const maxAttempts = 100;
    
    do {
      material = Math.floor(100000 + Math.random() * 900000).toString();
      attempts++;
    } while (window.usedMaterials.has(material) && attempts < maxAttempts);
    
    if (attempts >= maxAttempts) {
      // Fallback: use timestamp-based material
      material = Date.now().toString().slice(-6);
    }
    
    window.usedMaterials.add(material);
    return material;
  }

  generateRandomNetValue() {
    // Generate random value between 1 and 100 USD
    const value = (Math.random() * 99 + 1).toFixed(2);
    return value;
  }
}

/**
 * Inicialización de la página
 * NOTA: Este módulo se carga dinámicamente a través del router.
 * No usar DOMContentLoaded aquí porque ya habrá ocurrido cuando el router importe este archivo.
 * El router ejecuta automáticamente default() o init() después de inyectar el HTML.
 */
export function init() {
  new PDFExtractionPage();
}

// Export por defecto para compatibilidad con el router (llama default() si existe)
export default init;

// Exportar clase para uso externo si se necesita
export { PDFExtractionPage };
