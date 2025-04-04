import Input from 'sap/m/Input';
import MessageBox from 'sap/m/MessageBox';
import Text from 'sap/m/Text';
import Controller from 'sap/ui/core/mvc/Controller';
import ODataContextBinding from 'sap/ui/model/odata/v4/ODataContextBinding';
import AppComponent from '../Component';
import TextArea from 'sap/m/TextArea';
import CheckBox from 'sap/m/CheckBox';

/**
 * @namespace ui5.typescript.askemail.controller
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
			const sInputPrompt = (oView.byId('inputPrompt') as Input)
				.getValue()
				.trim();
			const bUseMasking = (oView.byId('useMasking') as CheckBox).getSelected();

			if (!sInputPrompt) {
				MessageBox.warning('Please enter a prompt');
				return;
			}

			// Show loading state
			const oResultText = oView.byId('resultText') as TextArea;
			oResultText.setValue('Loading...');
			oResultText.setVisible(true);

			// Create and execute the OData action
			const oBinding = oModel?.bindContext(
				'/generateEmail(...)'
			) as ODataContextBinding;
			oBinding.setParameter('prompt', sInputPrompt);
			oBinding.setParameter('anonymize', bUseMasking);
			await oBinding.invoke();

			// Show result
			const oResult: { value: string } = oBinding
				.getBoundContext()
				.getObject() as { value: string };
			oResultText.setValue(`Result: ${oResult?.value}`);
		} catch (error) {
			console.error(error);
			MessageBox.error('Could not retrieve response');
			(oView.byId('resultText') as Text).setVisible(false);
		}
	}
}
