/* Sales Order specific UI5 components */
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Link.js";
import "@ui5/webcomponents/dist/CheckBox.js";
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableHeaderRow.js";
import "@ui5/webcomponents/dist/TableHeaderCell.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";

class SalesOrderPage {
  constructor() {
    // Load data from sessionStorage if available
    this.loadSalesOrderData();
    
    this.initializeElements();
    this.attachEventListeners();
    this.loadData();
    
    console.log('SalesOrderPage initialized');
  }

  loadSalesOrderData() {
    // Try to load data from sessionStorage first
    const storedData = sessionStorage.getItem('salesOrderData');
    
    if (storedData) {
      try {
        const parsedData = JSON.parse(storedData);
        console.log('Loaded Sales Order data from sessionStorage:', parsedData);
        
        this.salesOrderData = {
          header: {
            salesDocument: "170",
            salesDocumentType: "Standard Order - OR (OR)",
            netValue: this.calculateTotalNetValue(parsedData.items),
            salesOrganization: "2510",
            distributionChannel: "10",
            division: "00",
            salesOffice: "",
            salesGroup: "",
            soldToParty: "25100004",
            shipToParty: "25100004",
            customerReference: "Generated from PDF",
            createdBy: "SAP AI",
            createdOn: new Date().toLocaleDateString('en-US', { 
              year: 'numeric', 
              month: 'short', 
              day: 'numeric' 
            }),
            time: new Date().toLocaleTimeString('en-US'),
            changedOn: new Date().toLocaleDateString('en-US', { 
              year: 'numeric', 
              month: 'short', 
              day: 'numeric' 
            })
          },
          items: parsedData.items || []
        };
        
        // Clear sessionStorage after loading
        sessionStorage.removeItem('salesOrderData');
        
      } catch (error) {
        console.error('Error parsing Sales Order data from sessionStorage:', error);
        this.loadDefaultData();
      }
    } else {
      this.loadDefaultData();
    }
  }

  loadDefaultData() {
    this.salesOrderData = {
      header: {
        salesDocument: "170",
        salesDocumentType: "Standard Order - OR (OR)",
        netValue: "280.80 USD",
        salesOrganization: "2510",
        distributionChannel: "10",
        division: "00",
        salesOffice: "",
        salesGroup: "",
        soldToParty: "25100004",
        shipToParty: "25100004",
        customerReference: "Test Fiori",
        createdBy: "SAP AI",
        createdOn: new Date().toLocaleDateString('en-US', { 
          year: 'numeric', 
          month: 'short', 
          day: 'numeric' 
        }),
        time: new Date().toLocaleTimeString('en-US'),
        changedOn: new Date().toLocaleDateString('en-US', { 
          year: 'numeric', 
          month: 'short', 
          day: 'numeric' 
        })
      },
      items: [
        {
          item: "10",
          material: "123456",
          description: "Sample Product",
          quantity: "1.000",
          unit: "PC",
          netValue: "50.00 USD",
          createdBy: "SAP AI",
          createdOn: new Date().toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
          }),
          createdAt: new Date().toLocaleTimeString('en-US')
        }
      ]
    };
  }

  calculateTotalNetValue(items) {
    if (!items || items.length === 0) return "0.00 USD";
    
    const total = items.reduce((sum, item) => {
      const value = parseFloat(item.netValue.replace(/[^\d.]/g, '')) || 0;
      return sum + value;
    }, 0);
    
    return `${total.toFixed(2)} USD`;
  }

  initializeElements() {
    // Get references to UI elements
    this.tabContainer = document.getElementById('main-tabs');
    this.itemsTable = document.getElementById('items-table');
    
    // Header form elements
    this.salesOrgInput = document.getElementById('sales-org');
    this.distChannelInput = document.getElementById('dist-channel');
    this.divisionInput = document.getElementById('division');
    this.salesOfficeInput = document.getElementById('sales-office');
    this.salesGroupInput = document.getElementById('sales-group');
    this.soldToPartyInput = document.getElementById('sold-to-party');
    this.shipToPartyInput = document.getElementById('ship-to-party');
    this.customerRefInput = document.getElementById('customer-ref');
  }

  attachEventListeners() {
    console.log('Attaching event listeners...');
    
    // Wait for UI5 components to be fully loaded
    this.waitForUI5Components().then(() => {
      this.setupEventListeners();
    });
  }

  async waitForUI5Components() {
    const maxAttempts = 50;
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      if (customElements.get('ui5-tabcontainer') && customElements.get('ui5-button')) {
        console.log('UI5 components are ready');
        return;
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }
    
    console.warn('UI5 components may not be fully loaded, proceeding anyway');
  }

  setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Save button
    const saveButton = document.querySelector('.footer-actions ui5-button[design="Emphasized"]');
    if (saveButton) {
      saveButton.addEventListener('click', () => {
        this.handleSave();
      });
    }

    // Cancel button
    const cancelButton = document.querySelector('.footer-actions ui5-button[design="Transparent"]');
    if (cancelButton) {
      cancelButton.addEventListener('click', () => {
        this.handleCancel();
      });
    }

    // Add item button
    const addButton = document.querySelector('ui5-button[design="Emphasized"] ui5-icon[name="add"]');
    if (addButton && addButton.parentElement) {
      addButton.parentElement.addEventListener('click', () => {
        this.handleAddItem();
      });
    }

    // Delete button
    const deleteButton = document.querySelector('ui5-button[design="Transparent"]:not(.footer-actions ui5-button)');
    if (deleteButton && deleteButton.textContent.trim() === 'Delete') {
      deleteButton.addEventListener('click', () => {
        this.handleDeleteItems();
      });
    }

    // Input change listeners for auto-save
    this.setupInputListeners();
    
    console.log('All event listeners attached successfully');
  }

  setupInputListeners() {
    const inputs = [
      this.salesOrgInput,
      this.distChannelInput,
      this.divisionInput,
      this.salesOfficeInput,
      this.salesGroupInput,
      this.soldToPartyInput,
      this.shipToPartyInput,
      this.customerRefInput
    ];

    inputs.forEach(input => {
      if (input) {
        input.addEventListener('change', () => {
          this.handleFieldChange();
        });
      }
    });
  }

  loadData() {
    console.log('Loading sales order data...');
    
    // Load header data is already set in HTML with values
    // This method could be used to load data from an API
    
    // Update items table if needed
    this.updateItemsTable();
  }

  updateItemsTable() {
    // Clear existing data rows (keep header)
    const existingRows = this.itemsTable.querySelectorAll('ui5-table-row');
    existingRows.forEach(row => row.remove());

    // Add items from data
    this.salesOrderData.items.forEach(item => {
      this.addItemToTable(item);
    });
  }

  addItemToTable(item) {
    const row = document.createElement('ui5-table-row');
    
    // Checkbox cell
    const checkboxCell = document.createElement('ui5-table-cell');
    const checkbox = document.createElement('ui5-checkbox');
    checkboxCell.appendChild(checkbox);
    row.appendChild(checkboxCell);

    // Item cell
    const itemCell = document.createElement('ui5-table-cell');
    const itemText = document.createElement('ui5-text');
    itemText.textContent = item.item;
    itemCell.appendChild(itemText);
    row.appendChild(itemCell);

    // Material cell
    const materialCell = document.createElement('ui5-table-cell');
    const materialLink = document.createElement('ui5-link');
    materialLink.textContent = item.material;
    const helpIcon = document.createElement('ui5-icon');
    helpIcon.name = 'value-help';
    helpIcon.style.marginLeft = '0.5rem';
    materialCell.appendChild(materialLink);
    materialCell.appendChild(helpIcon);
    row.appendChild(materialCell);

    // Description cell
    const descCell = document.createElement('ui5-table-cell');
    const descText = document.createElement('ui5-text');
    descText.textContent = item.description;
    descCell.appendChild(descText);
    row.appendChild(descCell);

    // Quantity cell
    const quantityCell = document.createElement('ui5-table-cell');
    const quantityText = document.createElement('ui5-text');
    quantityText.textContent = item.quantity;
    quantityText.style.color = '#d32f2f';
    quantityText.style.fontWeight = 'bold';
    const unitText = document.createElement('ui5-text');
    unitText.textContent = item.unit;
    unitText.style.marginLeft = '0.5rem';
    const quantityHelpIcon = document.createElement('ui5-icon');
    quantityHelpIcon.name = 'value-help';
    quantityHelpIcon.style.marginLeft = '0.5rem';
    quantityCell.appendChild(quantityText);
    quantityCell.appendChild(unitText);
    quantityCell.appendChild(quantityHelpIcon);
    row.appendChild(quantityCell);

    // Net Value cell
    const valueCell = document.createElement('ui5-table-cell');
    const valueText = document.createElement('ui5-text');
    valueText.textContent = item.netValue;
    valueText.style.color = '#d32f2f';
    valueText.style.fontWeight = 'bold';
    valueCell.appendChild(valueText);
    row.appendChild(valueCell);

    // Created By cell
    const createdByCell = document.createElement('ui5-table-cell');
    const createdByText = document.createElement('ui5-text');
    createdByText.textContent = item.createdBy;
    createdByCell.appendChild(createdByText);
    row.appendChild(createdByCell);

    // Created On cell
    const createdOnCell = document.createElement('ui5-table-cell');
    const createdOnText = document.createElement('ui5-text');
    createdOnText.textContent = item.createdOn;
    createdOnCell.appendChild(createdOnText);
    row.appendChild(createdOnCell);

    // Created At cell
    const createdAtCell = document.createElement('ui5-table-cell');
    const createdAtText = document.createElement('ui5-text');
    createdAtText.textContent = item.createdAt;
    createdAtCell.appendChild(createdAtText);
    row.appendChild(createdAtCell);

    this.itemsTable.appendChild(row);
  }

  async handleSave() {
    console.log('Save button clicked');
    
    try {
      // Collect form data
      const formData = this.collectFormData();
      
      console.log('Sending Sales Order data to backend:', formData);
      
      // Hacer llamada al backend para generar número de orden
      const response = await fetch('/api/sales-order/generate-number', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Show success message with Sales Order number
        this.showMessage(`Sales Order ${result.salesOrderNumber} generated successfully!`, 'Success');
        console.log('Sales Order number generated:', result.salesOrderNumber);
        
        // Actualizar el campo de número de documento en la UI si existe
        const salesDocumentField = document.getElementById('sales-document');
        if (salesDocumentField) {
          salesDocumentField.value = result.salesOrderNumber;
        }
      } else {
        throw new Error(result.message || 'Unknown error occurred');
      }
      
    } catch (error) {
      console.error('Error saving Sales Order:', error);
      this.showMessage(`Error saving Sales Order: ${error.message}`, 'Error');
    }
  }

  generateSalesOrderNumber() {
    // Get current counter from localStorage, starting at 1000
    let currentNumber = parseInt(localStorage.getItem('salesOrderCounter') || '1000');
    
    // Increment for next use
    localStorage.setItem('salesOrderCounter', (currentNumber + 1).toString());
    
    return currentNumber.toString();
  }

  handleCancel() {
    console.log('Cancel button clicked');
    
    // Here you would typically navigate back or reset the form
    if (confirm('Are you sure you want to cancel? Any unsaved changes will be lost.')) {
      // Reset form or navigate away
      this.resetForm();
    }
  }

  handleAddItem() {
    console.log('Add item button clicked');
    
    // Create a new item with default values
    const newItem = {
      item: (this.salesOrderData.items.length + 1) * 10,
      material: "",
      description: "",
      quantity: "1.000",
      unit: "PC",
      netValue: "0.00 EUR",
      createdBy: "GJKLAPS",
      createdOn: new Date().toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      }),
      createdAt: new Date().toLocaleTimeString('en-US'),
      changedOn: new Date().toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      })
    };
    
    // Add to data
    this.salesOrderData.items.push(newItem);
    
    // Add to table
    this.addItemToTable(newItem);
    
    // Update items count
    this.updateItemsCount();
  }

  handleDeleteItems() {
    console.log('Delete items button clicked');
    
    // Get selected checkboxes
    const checkboxes = this.itemsTable.querySelectorAll('ui5-checkbox');
    const selectedIndices = [];
    
    checkboxes.forEach((checkbox, index) => {
      if (checkbox.checked) {
        selectedIndices.push(index);
      }
    });
    
    if (selectedIndices.length === 0) {
      alert('Please select items to delete.');
      return;
    }
    
    if (confirm(`Are you sure you want to delete ${selectedIndices.length} item(s)?`)) {
      // Remove from data (in reverse order to maintain indices)
      selectedIndices.reverse().forEach(index => {
        this.salesOrderData.items.splice(index, 1);
      });
      
      // Refresh table
      this.updateItemsTable();
      this.updateItemsCount();
    }
  }

  handleFieldChange() {
    console.log('Field changed - auto-saving...');
    
    // Update draft saved indicator
    const draftText = document.querySelector('.footer-actions ui5-text');
    if (draftText) {
      draftText.textContent = 'Draft saved';
    }
  }

  collectFormData() {
    return {
      header: {
        salesOrganization: this.salesOrgInput?.value || '',
        distributionChannel: this.distChannelInput?.value || '',
        division: this.divisionInput?.value || '',
        salesOffice: this.salesOfficeInput?.value || '',
        salesGroup: this.salesGroupInput?.value || '',
        soldToParty: this.soldToPartyInput?.value || '',
        shipToParty: this.shipToPartyInput?.value || '',
        customerReference: this.customerRefInput?.value || ''
      },
      items: this.salesOrderData.items
    };
  }

  resetForm() {
    // Reset all input fields to their original values
    if (this.salesOrgInput) this.salesOrgInput.value = this.salesOrderData.header.salesOrganization;
    if (this.distChannelInput) this.distChannelInput.value = this.salesOrderData.header.distributionChannel;
    if (this.divisionInput) this.divisionInput.value = this.salesOrderData.header.division;
    if (this.salesOfficeInput) this.salesOfficeInput.value = this.salesOrderData.header.salesOffice;
    if (this.salesGroupInput) this.salesGroupInput.value = this.salesOrderData.header.salesGroup;
    if (this.soldToPartyInput) this.soldToPartyInput.value = this.salesOrderData.header.soldToParty;
    if (this.shipToPartyInput) this.shipToPartyInput.value = this.salesOrderData.header.shipToParty;
    if (this.customerRefInput) this.customerRefInput.value = this.salesOrderData.header.customerReference;
  }

  updateItemsCount() {
    const itemsTitle = document.querySelector('ui5-title[level="H5"]');
    if (itemsTitle) {
      itemsTitle.textContent = `Sales Order Items (${this.salesOrderData.items.length})`;
    }
  }

  showMessage(message, type) {
    // Create a temporary message
    console.log(`${type}: ${message}`);
    
    // You could implement a toast notification here
    alert(message);
  }
}

/**
 * Inicialización de la página
 */
export function init() {
  new SalesOrderPage();
}

// Export por defecto para compatibilidad con el router
export default init;

// Exportar clase para uso externo si se necesita
export { SalesOrderPage };
