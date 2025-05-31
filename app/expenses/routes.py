from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from ..models import Expense
from .. import db
from .ml_models import predict_category, predict_next_month_by_category, aggregate_expenses_by_month
from .forms import AddExpenseForm
from datetime import datetime

expenses = Blueprint('expenses', __name__, template_folder='../templates/expenses')

@expenses.route('/')
@login_required
def dashboard():
    user_expenses = Expense.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', expenses=user_expenses)

@expenses.route('/add', methods=['GET', 'POST'])
@login_required
def add_expense():
    form = AddExpenseForm()
    if form.validate_on_submit():
        date = form.date.data
        description = form.description.data
        amount = float(form.amount.data)

        # Server-side validation:
        if amount <= 0:
            flash('Amount must be greater than ₹0.', 'danger')
            return redirect(url_for('expenses.add_expense'))
        if amount > 1e7:
            flash('Amount cannot exceed ₹1 crore.', 'danger')
            return redirect(url_for('expenses.add_expense'))

        category = predict_category(description)

        new_expense = Expense(
            user_id=current_user.id,
            date=date,
            description=description,
            amount=amount,
            category=category
        )
        db.session.add(new_expense)
        db.session.commit()

        flash('Expense added!', 'success')
        return redirect(url_for('expenses.dashboard'))

    return render_template('add_expense.html', form=form, today=datetime.utcnow().strftime('%Y-%m-%d'))

@expenses.route('/predict')
@login_required
def predict():
    user_expenses = Expense.query.filter_by(user_id=current_user.id).all()
    category_predictions = predict_next_month_by_category(user_expenses)
    monthly_df = aggregate_expenses_by_month(user_expenses)
    monthly_data = monthly_df.to_dict(orient='records') if not monthly_df.empty else []

    return render_template(
        'predict.html',
        category_predictions=category_predictions,
        monthly_data=monthly_data
    )

@expenses.route('/expense_trends')
@login_required
def expense_trends():
    user_expenses = Expense.query.filter_by(user_id=current_user.id).all()
    monthly_df = aggregate_expenses_by_month(user_expenses)
    monthly_data = monthly_df.to_dict(orient='records') if not monthly_df.empty else []

    return render_template('trend_analysis.html', monthly_data=monthly_data)

