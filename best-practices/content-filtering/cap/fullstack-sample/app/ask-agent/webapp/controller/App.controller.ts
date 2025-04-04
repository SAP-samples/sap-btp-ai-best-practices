import Input from 'sap/m/Input';
import MessageBox from 'sap/m/MessageBox';
import Text from 'sap/m/Text';
import Controller from 'sap/ui/core/mvc/Controller';
import ODataContextBinding from 'sap/ui/model/odata/v4/ODataContextBinding';
import AppComponent from '../Component';
import CheckBox from 'sap/m/CheckBox';
import TextArea from 'sap/m/TextArea';

/**
 * @namespace ui5.typescript.askagent.controller
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
			const bFilterInput = (
				oView.byId('filterInput') as CheckBox
			).getSelected();

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
				'/chatWithAgent(...)'
			) as ODataContextBinding;
			oBinding.setParameter('input', sInputPrompt);
			oBinding.setParameter('filterInput', bFilterInput);
			await oBinding.invoke();

			// Show result
			const oResult: { value: string } = oBinding
				.getBoundContext()
				.getObject() as { value: string };
			oResultText.setValue(`Result: ${oResult?.value}`);
		} catch (error: unknown) {
			if (error instanceof Error) {
				console.error(error.message);
				MessageBox.error(`Could not retrieve response - ${error.message}`);
			}
			(oView.byId('resultText') as Text).setVisible(false);
		}
	}

	public async onSubmit2() {
		const oView = this.getView()!;
		try {
			const oModel = oView.getModel();
			const sInputPrompt2 = (oView.byId('inputPrompt2') as Input)
				.getValue()
				.trim();
			const bFilterOutput = (
				oView.byId('filterOutput') as CheckBox
			).getSelected();

			if (!sInputPrompt2) {
				MessageBox.warning('Please enter a prompt');
				return;
			}

			// Show loading state
			const oResultText = oView.byId('resultText2') as TextArea;
			oResultText.setValue('Loading...');
			oResultText.setVisible(true);

			// Create and execute the OData action
			const oBinding = oModel?.bindContext(
				'/generateParaphrase(...)'
			) as ODataContextBinding;
			oBinding.setParameter('input', sInputPrompt2);
			oBinding.setParameter('filterOutput', bFilterOutput);
			await oBinding.invoke();

			// Show result
			const oResult: { value: string } = oBinding
				.getBoundContext()
				.getObject() as { value: string };
			oResultText.setValue(`Result: ${oResult?.value}`);
		} catch (error: unknown) {
			if (error instanceof Error) {
				console.error(error.message);
				MessageBox.error(`Could not retrieve response - ${error.message}`);
			}
			(oView.byId('resultText2') as Text).setVisible(false);
		}
	}
}
