from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from ..models import Expense
from .. import db
from .ml_models import predict_category, predict_future_expense
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
        flash('Expense added!')
        return redirect(url_for('expenses.dashboard'))
    return render_template('add_expense.html', form=form)

@expenses.route('/predict')
@login_required
def predict():
    # Query all expenses for the current user from the database
    user_expenses = Expense.query.filter_by(user_id=current_user.id).all()
    # Pass the list of expenses to the prediction function
    future_expense = predict_future_expense(user_expenses)
    return render_template('predict.html', future_expense=future_expense)

