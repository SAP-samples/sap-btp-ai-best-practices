import Input from 'sap/m/Input';
import MessageBox from 'sap/m/MessageBox';
import Text from 'sap/m/Text';
import Controller from 'sap/ui/core/mvc/Controller';
import ODataContextBinding from 'sap/ui/model/odata/v4/ODataContextBinding';
import AppComponent from '../Component';

/**
 * @namespace ui5.typescript.askcapital.controller
 */
export default class App extends Controller {
	public onInit(): void {
		// apply content density mode to root view
		const oView = this.getView();
		if (oView) {
			oView.addStyleClass(
				(this.getOwnerComponent() as AppComponent).getContentDensityClass()
			);
		}
	}

	public async onSubmit() {
		const oView = this.getView()!;
		try {
			const oModel = oView.getModel();
			const sCountry = (oView.byId('inputCountry') as Input).getValue()?.trim();

			if (!sCountry) {
				MessageBox.warning('Please enter a country name');
				return;
			}

			// Show loading state
			const oResultText = oView.byId('resultText') as Text;
			oResultText.setText('Loading...');
			oResultText.setVisible(true);

			// Create and execute the OData action
			const oBinding = oModel?.bindContext(
				'/askCapitalOfCountry(...)'
			) as ODataContextBinding;
			oBinding.setParameter('country', sCountry);
			await oBinding.invoke();

			// Show result
			const oResult: { value: string } = oBinding
				.getBoundContext()
				.getObject() as { value: string };
			oResultText.setText(`Result: ${oResult.value}`);
		} catch (error) {
			console.error(error);
			MessageBox.error('Could not retrieve capital information');
			(oView.byId('resultText') as Text).setVisible(false);
		}
	}
}
